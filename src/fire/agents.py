import networkx as nx
from mesa import Agent

from .board import Board, CellState, Direction, PoiState, WallState


STARTING_AP = 4
MAX_SAVED_AP = 4
MOVE_COST = 1
MOVE_INTO_FIRE_COST = 2
CARRY_VICTIM_COST = 2
CARRY_VICTIM_INTO_FIRE_COST = 3  # extinguir (1) + mover con víctima (2). Podría ser 4 por seguridad
DOOR_COST = 1
CHOP_COST = 2
EXTINGUISH_SMOKE_COST = 1
EXTINGUISH_FIRE_COST = 2
CHOP_DAMAGE_LIMIT = 15

_FIRE_CLUSTER_RADIUS = 2

_MAX_ACTIONS_PER_TURN = 20

RESCUE_ROLE = "rescue"
SUPPRESSION_ROLE = "fire_suppression"
EVACUATION_ROLE = "evacuation"

class Firefighter(Agent):
    """Un bombero con diferentes roles:
      - 'rescue':  va camino a su víctima reclamada (POI aún sin revelar,
                   o revelado pero lejos).
      - 'evacuation': está sobre su víctima revelada y la saca al exterior.
      - 'fire_suppression': apaga fuego y puede reclamar POI nuevos.
    El modelo reasigna solo las víctimas y agentes sin reclamo; un POI
    reclamado debe ser revelado y evacuado por su dueño, nunca se le
    quita en medio."""

    def __init__(self, model, pos: tuple[int, int]):
        super().__init__(model)
        self.pos = pos
        self.ap = 0
        self.saved_ap = 0
        self.knocked_down = False
        self.action_summary: list[str] = []
        self.ap_spent: int = 0
        self.turn_start_pos: tuple[int, int] = pos

        self.role: str = "fire_suppression"
        self.claimed_victim: tuple[int, int] | None = None
        self.target: tuple[int, int] | None = None
        self.path: list[tuple[int, int]] | None = None

    @property
    def board(self) -> Board:
        return self.model.board

    # ---- Weighted graph (Dijkstra) -------------------------------------

    def _weighted_graph(
        self, carrying: bool = False, fire_cost: int | None = None, allow_chop: bool = True
    ) -> nx.DiGraph:
        """Grafo dirigido con costos de AP solo en aristas.

        Cada arista (u→v) suma:
          1. Costo estructural del borde (0=open, DOOR_COST, o CHOP_COSTs).
          2. Costo de pisar la celda destino (0 si es la posición actual,
             fire_cost si hay fuego, MOVE_COST en caso contrario).

        Al tener todos los costos en aristas, Dijkstra encuentra directamente
        el camino mínimo sin necesidad de recálculo manual."""
        if fire_cost is None:
            # fire_cost = EXTINGUISH_FIRE_COST + (
            #     CARRY_VICTIM_INTO_FIRE_COST if carrying else MOVE_COST
            # )
            fire_cost = CARRY_VICTIM_INTO_FIRE_COST if carrying else MOVE_COST

        graph = nx.DiGraph()

        # Un evacuador no puede pisar una celda que ya tenga otro POI: al
        # mover la víctima ahí se apilarían dos marcadores en la misma
        # casilla. Solo se le permite su propia celda (la de su víctima).
        blocked = {
            pos for pos in self.board.G.nodes()
            if carrying and self.board.has_poi(pos) and pos != self.pos
        }

        for pos in self.board.G.nodes():
            if pos not in blocked:
                graph.add_node(pos)

        def _enter_cost(pos: tuple[int, int]) -> int:
            if pos == self.pos:
                return 0
            if self.board.get_state(pos) == CellState.FIRE:
                return fire_cost
            return MOVE_COST

        for u, v, data in self.board.G.edges(data=True):
            if u in blocked or v in blocked:
                continue
            wall = data["wall"]
            if wall in (WallState.CLEAR, WallState.DOOR_OPEN):
                structural = 0
            elif wall == WallState.DOOR_CLOSE:
                structural = DOOR_COST
            elif wall in (WallState.WALL, WallState.DAMAGED_WALL) and allow_chop:
                if self.model.damage_total >= CHOP_DAMAGE_LIMIT:
                    continue
                chop_count = 2 if wall == WallState.WALL else 1
                structural = chop_count * CHOP_COST
            else:
                continue

            graph.add_edge(u, v, weight=structural + _enter_cost(v))
            graph.add_edge(v, u, weight=structural + _enter_cost(u))

        return graph

    def _weighted_path_to(
        self,
        targets: set[tuple[int, int]],
        carrying: bool = False,
        fire_cost: int | None = None,
        allow_chop: bool = True,
    ) -> list[tuple[int, int]] | None:
        """Camino de menor costo (Dijkstra) desde la posición actual
        hasta cualquiera de los targets usando pesos de AP reales."""
        if not targets:
            return None
        graph = self._weighted_graph(carrying=carrying, fire_cost=fire_cost, allow_chop=allow_chop)
        if self.pos not in graph:
            return None
        return self._best_weighted_path(graph, targets)

    def _best_weighted_path(
        self, graph: nx.DiGraph, targets: set[tuple[int, int]]
    ) -> list[tuple[int, int]] | None:
        """Camino de menor costo hacia el target más cercano.

        Ejecuta Dijkstra una sola vez desde self.pos y elige el target
        alcanzable de menor costo. Todos los costos viven en las aristas,
        así que el costo del camino es directamente el de Dijkstra."""
        try:
            lengths, paths = nx.single_source_dijkstra(graph, self.pos, weight="weight")
        except nx.NetworkXNoPath:
            return None
        reachable = [t for t in targets if t in lengths]
        if not reachable:
            return None
        best_target = min(reachable, key=lambda t: lengths[t])
        return paths[best_target]

    # ---- helpers ------------------------------------------------------

    def _neighbor(self, direction: Direction) -> tuple[int, int] | None:
        return self.board.neighbor(self.pos, direction)

    def _spend(self, cost: int) -> bool:
        if self.ap < cost:
            return False
        self.ap -= cost
        self.ap_spent += cost
        return True

    def _direction_to(self, target: tuple[int, int]) -> Direction | None:
        for direction in Direction:
            if self.board.neighbor(self.pos, direction) == target:
                return direction
        return None

    def _exterior_cells(self) -> set[tuple[int, int]]:
        return {
            (r, c)
            for r in range(self.board.height)
            for c in range(self.board.width)
            if self.board.is_outside((r, c))
        }

    def _fire_smoke_cells(self) -> set[tuple[int, int]]:
        return {
            pos
            for pos, data in self.board.G.nodes(data=True)
            if data["state"] in (CellState.FIRE, CellState.SMOKE)
        }

    # ---- game actions -------------------------------------------------

    def move(self, direction: Direction, carry: bool = False) -> bool:
        target = self._neighbor(direction)
        if target is None or not self.board.is_passable(self.pos, direction):
            return False

        carrying_now = carry or self.role == EVACUATION_ROLE
        if carrying_now:
            if self.board.has_poi(target):
                # La celda destino ya tiene otro POI (víctima o falsa
                # alarma): no se puede apilar la víctima encima.
                return False
            poi = self.board.poi_at(self.pos)
            if (
                poi is None
                or poi.kind != PoiState.VICTIM
                or not poi.revealed
                or poi.claim is not self
            ):
                carrying_now = False

        state = self.board.get_state(target)
        if carrying_now and state == CellState.FIRE:
            cost = CARRY_VICTIM_INTO_FIRE_COST
        elif carrying_now:
            cost = CARRY_VICTIM_COST
        elif state == CellState.FIRE:
            cost = MOVE_INTO_FIRE_COST
        else:
            cost = MOVE_COST

        if not self._spend(cost):
            return False

        old_pos = self.pos
        self.pos = target
        self.action_summary.append(f"move:{direction.name}")

        if carrying_now:
            self.board.move_victim(old_pos, target)
            self.claimed_victim = target
            if self.board.is_outside(target):
                self.model.rescue_victim(target)
                self.action_summary.append("rescue")
        elif self.board.has_poi(target):
            poi = self.board.poi_at(target)
            if poi is not None and not poi.revealed:
                kind = self.board.reveal_poi(target)
                self.action_summary.append(f"reveal:{kind.name}")
                if kind == PoiState.FALSE:
                    self.model.identify_false_alarm(target)
        return True

    def open_close_door(self, direction: Direction) -> bool:
        wall = self.board.get_wall(self.pos, direction)
        if wall not in (WallState.DOOR_OPEN, WallState.DOOR_CLOSE):
            return False
        if not self._spend(DOOR_COST):
            return False
        new_state = WallState.DOOR_CLOSE if wall == WallState.DOOR_OPEN else WallState.DOOR_OPEN
        self.board.set_wall(self.pos, direction, new_state)
        self.action_summary.append(f"door:{direction.name}")
        return True

    def extinguish(self, direction: Direction | None = None) -> bool:
        target = self.pos if direction is None else self._neighbor(direction)
        if target is None:
            return False
        state = self.board.get_state(target)
        if state == CellState.SMOKE:
            if not self._spend(EXTINGUISH_SMOKE_COST):
                return False
            self.board.set_state(target, CellState.CLEAR)
            self.action_summary.append(f"extinguish:SMOKE@{target}")
            return True
        if state == CellState.FIRE:
            if not self._spend(EXTINGUISH_FIRE_COST):
                return False
            self.board.set_state(target, CellState.CLEAR)
            self.action_summary.append(f"extinguish:FIRE@{target}")
            return True
        return False

    def chop(self, direction: Direction) -> bool:
        if self.model.damage_total >= CHOP_DAMAGE_LIMIT:
            return False
        wall = self.board.get_wall(self.pos, direction)
        if wall not in (WallState.WALL, WallState.DAMAGED_WALL):
            return False
        if not self._spend(CHOP_COST):
            return False
        new_state = WallState.CLEAR if wall == WallState.DAMAGED_WALL else WallState.DAMAGED_WALL
        self.board.set_wall(self.pos, direction, new_state)
        self.model.register_damage(1)
        self.action_summary.append(f"chop:{direction.name}")
        return True

    # ---- turn ---------------------------------------------------------

    def take_turn(self):
        if self.model.status != "in_progress":
            return

        self.knocked_down = False
        self.action_summary = []
        self.ap_spent = 0
        self.turn_start_pos = self.pos  # type: ignore[assignment]

        self.ap = STARTING_AP + self.saved_ap
        self.saved_ap = 0

        self.path = None
        self._act()

        self.saved_ap = min(self.ap, MAX_SAVED_AP)
        self.ap = 0

    # ---- strategy -----------------------------------------------------

    def _act(self):
        for _ in range(_MAX_ACTIONS_PER_TURN):
            if self.ap <= 0 or self.model.status != "in_progress":
                return

            self._refresh_role()

            # Un rescatista no pierde tiempo apagando fuegos ajenos fuera
            # de su ruta: solo limpia su propia celda y el fuego que le
            # bloquea el paso (eso ya lo hace el paso por el camino).
            if self.role == SUPPRESSION_ROLE:
                if self.extinguish(None):
                    continue
                if self._extinguish_fire_a_neighbor():
                    continue
                if self._extinguish_smoke_a_neighbor():
                    continue

            poi_here = self.board.poi_at(self.pos)
            if poi_here is not None and not poi_here.revealed:
                kind = self.board.reveal_poi(self.pos)
                self.action_summary.append(f"reveal:{kind.name}")
                poi_here = self.board.poi_at(self.pos)
                if poi_here is None:
                    self.model.identify_false_alarm(self.pos)

            # Revelar pudo cambiarnos de rol (ya estamos sobre nuestra
            # víctima). Re-sincronizamos antes de elegir función de mov.
            self._refresh_role()

            if self.role == RESCUE_ROLE:
                if not self._move_to_poi():
                    return
            elif self.role == EVACUATION_ROLE:
                if not self._get_victim_out():
                    return
            else:
                if not self._step_suppression():
                    return

    def _extinguish_fire_a_neighbor(self, own: bool = True) -> bool:
        """Primero todo lo que esté en llamas: propia celda y vecinos."""
        if own and self.board.get_state(self.pos) == CellState.FIRE:
            if self.extinguish(None):
                return True
        for direction in Direction:
            neighbor = self._neighbor(direction)
            if (
                neighbor is not None
                and self.board.get_state(neighbor) == CellState.FIRE
                and self.extinguish(direction)
            ):
                return True
        return False

    def _extinguish_smoke_a_neighbor(self, own: bool = True) -> bool:
        """Luego humo (propia celda y vecinos): apagar fuego SIEMPRE manda."""
        if own and self.board.get_state(self.pos) == CellState.SMOKE:
            if self.extinguish(None):
                return True
        for direction in Direction:
            neighbor = self._neighbor(direction)
            if (
                neighbor is not None
                and self.board.get_state(neighbor) == CellState.SMOKE
                and self.extinguish(direction)
            ):
                return True
        return False   

    def _refresh_role(self) -> None:
        """Deriva el rol a partir del reclamo y de la posición:
        sin víctima reclamada -> supresión; sobre la propia víctima
        revelada -> evacuación; cualquier otro reclamo vigente -> rescue.
        Si el POI reclamado desapareció, se suelta el reclamo."""
        if self.claimed_victim is None:
            self.role = SUPPRESSION_ROLE
            return
        poi = self.board.poi_at(self.claimed_victim)
        if poi is None or poi.claim is not self:
            self.claimed_victim = None
            self.role = SUPPRESSION_ROLE
            return
        if poi.revealed and poi.pos == self.pos:
            self.role = EVACUATION_ROLE
        else:
            self.role = RESCUE_ROLE

    def _move_to_poi(self) -> bool:
        """Un rescatista va hacia su víctima reclamada (todavía no la
        carga). Camina con costo de incendio normal de rescate y sin
        cargar a nadie."""
        victim = self.claimed_victim
        if victim is None:
            return False
        self.path = self._weighted_path_to({victim}, carrying=False)
        if not self.path or len(self.path) < 2:
            return False
        return self._step_once(self.path[1], carry=False, allow_chop=True)

    def _get_victim_out(self) -> bool:
        """Un evacuador saca a la víctima que transporta hasta cualquier
        celda del exterior."""
        self.path = self._weighted_path_to(
            self._exterior_cells(), carrying=True, fire_cost=None
        )
        if not self.path or len(self.path) < 2:
            return False
        return self._step_once(self.path[1], carry=True, allow_chop=True)

    def _step_suppression(self) -> bool:
        """Un bombero de supresión recomputa su camino al foco (sin
        hachar paredes) y avanza una celda."""
        self._fire_suppression_strategy()
        if not self.path or len(self.path) < 2:
            return False
        return self._step_once(self.path[1], carry=False, allow_chop=False)

    def _fire_suppression_strategy(self):
        fire_cells = self._fire_smoke_cells()
        if not fire_cells:
            self.path = None
            return

        # El modelo ya asignó un objetivo distinto a cada agente de
        # supresión (para que no se amontonen en el mismo foco). Si por
        # lo que sea el objetivo desapareció, recurrimos al foco más
        # denso disponible.
        target = self.target
        if target is None or self.board.get_state(target) not in (CellState.FIRE, CellState.SMOKE):
            target = self._choose_best_fire_target(fire_cells)
            self.target = target

        if self.board.get_state(target) == CellState.FIRE:
            approach = {target} | {
                n for d in Direction if (n := self.board.neighbor(target, d)) is not None
            }
        if self.board.get_state(target) == CellState.FIRE:
            approach = {target}

            for direction in Direction:
                neighbor = self.board.neighbor(target, direction)

                if neighbor is not None:
                    approach.add(neighbor)
        else:
            approach = {target}

        # Un bombero de supresión no hacha paredes: cada hachazo suma
        # daño estructural, y su trabajo es justamente evitar que el
        # edificio colapse. Solo usa puertas y pasillos libres.
        self.path = self._weighted_path_to(approach, carrying=False, allow_chop=False)

    def _choose_best_fire_target(self, fire_cells: set[tuple[int, int]]) -> tuple[int, int]:
        best_pos = None
        best_score = -1
        for pos in fire_cells:
            score = 0
            for dr in range(-_FIRE_CLUSTER_RADIUS, _FIRE_CLUSTER_RADIUS + 1):
                for dc in range(-_FIRE_CLUSTER_RADIUS, _FIRE_CLUSTER_RADIUS + 1):
                    check = (pos[0] + dr, pos[1] + dc)
                    if self.board.in_bounds(check) and self.board.get_state(check) in (CellState.FIRE, CellState.SMOKE):
                        score += 1
            wall_risk = sum(
                1
                for d in Direction
                if self.board.get_wall(pos, d) in (WallState.WALL, WallState.DAMAGED_WALL)
            )
            score += 2 * wall_risk
            if score > best_score:
                best_score = score
                best_pos = pos
        if best_pos is None:
            return min(fire_cells, key=lambda p: abs(p[0] - self.pos[0]) + abs(p[1] - self.pos[1]))
        return best_pos

    def _step_once(self, next_pos: tuple[int, int], carry: bool = False, allow_chop: bool = True) -> bool:
        """Avanza una sola celda hacia `next_pos`, resolviendo lo que
        haya en el borde/celda: hacha (si está permitido), abre la
        puerta, apaga el fuego bloqueante, o se mueve (cargando o no a
        su víctima). Devuelve False si no se pudo hacer nada útil."""
        direction = self._direction_to(next_pos)
        if direction is None:
            self.path = None
            return False

        wall = self.board.get_wall(self.pos, direction)
        state = self.board.get_state(next_pos)

        if wall in (WallState.WALL, WallState.DAMAGED_WALL):
            if not allow_chop:
                self.path = None
                return False
            if self.chop(direction):
                self.path = None
                return True
            return False

        if wall == WallState.DOOR_CLOSE:
            if self.open_close_door(direction):
                self.path = None
                return True
            return False

        if state == CellState.FIRE:
            own_state = self.board.get_state(self.pos)
            if own_state in (CellState.FIRE, CellState.SMOKE):
                self.extinguish(None)
                return True
            for d in Direction:
                if self._neighbor(d) == next_pos:
                    if self.extinguish(d):
                        return True
            return False

        if self.move(direction, carry=carry):
            self.path = None
            return True

        return False
