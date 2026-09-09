from mesa import Model

from .agents import EVACUATION_ROLE, RESCUE_ROLE, SUPPRESSION_ROLE, Firefighter
from .board import INITIAL_POI_POSITIONS, CellState, Direction, PoiState, WallState, create_board
from .data_collector import FireDataCollector
from .fire_rules import FireEvent, advance_fire, replenish_poi

TOTAL_VICTIMS = 10
TOTAL_FALSE_ALARMS = 5
POI_TARGET_ON_BOARD = 3

VICTIMS_TO_WIN = 7
VICTIMS_LOST_TO_LOSE = 4
COLLAPSE_DAMAGE = 24

HEALING_ZONES = [
    (0, 6),
    (3, 0),
    (4, 9),
    (7, 3),
]


class FirefighterModel(Model):

    def __init__(self):
        super().__init__()

        self.board = create_board()

        self.victim_pool = TOTAL_VICTIMS
        self.false_alarm_pool = TOTAL_FALSE_ALARMS
        for pos in INITIAL_POI_POSITIONS:
            kind = self._draw_poi_kind()
            if kind is not None:
                self.board.add_poi(pos, kind)

        self.damage_total = 0
        self.victims_rescued = 0
        self.victims_lost = 0
        self.false_alarms_found = 0
        self.turn = 0
        self.status = "in_progress"

        self.firefighters: list[Firefighter] = []
        self._spawn_firefighters()

        self._assign_victims()
        self._assign_fire_targets()

        self.collector = FireDataCollector()

    # ---- preparation --------------------------------------------------

    def _spawn_firefighters(self) -> None:
        """Coloca 6 bomberos afuera: 2 abajo, 2 derecha, 2 arriba."""
        positions = [
            (7, 3), (7, 4),    # abajo, frente a la puerta de (6,3)
            (3, 9), (4, 9),    # derecha, frente a la puerta de (4,8)
            (0, 5), (0, 6),    # arriba, frente a la puerta de (1,6)
        ]
        for i, pos in enumerate(positions):
            ff = Firefighter(self, pos)
            ff.role = SUPPRESSION_ROLE
            self.firefighters.append(ff)

    def _assign_victims(self) -> None:
        """Asigna víctimas a los bomberos libres usando greedy matching
        por distancia Manhattan, PERO conservando todos los reclamos
        vigentes: una víctima ya reclamada sigue siendo del mismo bombero
        hasta rescatarse o perderse (debe revelarla y evacuarla él).
        Solo se miran las víctimas sin dueño y los bomberos sin reclamo,
        así que en la práctica se reasigna únicamente "lo nuevo". Al
        final se derivan los roles a partir de (reclamo, posición)."""
        # 1) Conservar reclamos vigentes; soltar los que quedaron huérfanos.
        for ff in self.firefighters:
            if ff.claimed_victim is None:
                continue
            poi = self.board.poi_at(ff.claimed_victim)
            if poi is None or poi.kind != PoiState.VICTIM or poi.claim is not ff:
                ff.claimed_victim = None
                ff.role = SUPPRESSION_ROLE
            else:
                poi.claim = ff  # mantener el puntero de propiedad al día

        # 2) Víctimas reclamables: cualquier VICTIM sin dueño (revelada o
        #    no). Un bombero sin reclamo puede adjudicársela.
        claimable = [p for p in self.board.pois if p.kind == PoiState.VICTIM and p.claim is None]
        free = [ff for ff in self.firefighters if ff.claimed_victim is None]

        for _ in range(min(len(free), len(claimable), 3)):
            best_dist = float("inf")
            best_pair = None
            for ff in free:
                if ff.claimed_victim is not None:
                    continue
                for p in claimable:
                    if p.claim is not None:
                        continue
                    d = abs(ff.pos[0] - p.pos[0]) + abs(ff.pos[1] - p.pos[1])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (ff, p)
            if best_pair is None:
                break
            ff, p = best_pair
            ff.claimed_victim = p.pos
            p.claim = ff
            if p.revealed and p.pos == ff.pos:
                ff.role = EVACUATION_ROLE
            else:
                ff.role = RESCUE_ROLE

        # 3) Derivar el rol de todos a partir de su reclamo.
        for ff in self.firefighters:
            if ff.claimed_victim is None:
                ff.role = SUPPRESSION_ROLE
                continue
            poi = self.board.poi_at(ff.claimed_victim)
            if poi is None or poi.claim is not ff:
                ff.claimed_victim = None
                ff.role = SUPPRESSION_ROLE
            elif poi.revealed and poi.pos == ff.pos:
                ff.role = EVACUATION_ROLE
            else:
                ff.role = RESCUE_ROLE

    def _assign_fire_targets(self) -> None:
        """Asigna a cada bombero de supresión un foco de incendio
        distinto, para que no se amontonen todos en el mismo. Los que ya
        tienen un objetivo válido lo conservan (para no ir de un lado a
        otro cambiando de idea); solo se buscan objetivos para los
        agentes sin ninguno o cuyo objetivo ya desapareció. El puntaje de
        cada celda en llamas combina la densidad de fuego/humo a su
        alrededor con la cercanía a los POI (proteger los caminos hacia
        las víctimas)."""
        fire_cells = [
            pos for pos, data in self.board.G.nodes(data=True)
            if data["state"] == CellState.FIRE
        ]
        poi_positions = [p.pos for p in self.board.pois if not p.revealed]
        suppressors = [
            ff for ff in self.firefighters if ff.role == SUPPRESSION_ROLE
        ]

        if not fire_cells or not suppressors:
            for ff in suppressors:
                ff.target = None
            return

        scored = []
        for pos in fire_cells:
            score = 0
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    check = (pos[0] + dr, pos[1] + dc)
                    if (
                        self.board.in_bounds(check)
                        and self.board.get_state(check) in (CellState.FIRE, CellState.SMOKE)
                    ):
                        score += 1
            if poi_positions:
                score += 3 / (1 + min(
                    abs(pos[0] - p[0]) + abs(pos[1] - p[1]) for p in poi_positions
                ))
            wall_risk = sum(
                1
                for d in Direction
                if self.board.get_wall(pos, d) in (WallState.WALL, WallState.DAMAGED_WALL)
            )
            score += 2 * wall_risk
            scored.append((score, pos))
        scored.sort(key=lambda x: -x[0])

        needs_target = [ff for ff in suppressors if ff.target is None or self.board.get_state(ff.target) not in (CellState.FIRE, CellState.SMOKE)]
        taken = [ff.target for ff in suppressors if ff.target is not None and ff.target in {p for _, p in scored}]
        for ff in sorted(
            needs_target,
            key=lambda f: min((abs(f.pos[0] - p[1][0]) + abs(f.pos[1] - p[1][1]) for p in scored), default=0),
        ):
            pick = None
            for _, pos in scored:
                if all(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) > 1 for t in taken):
                    pick = pos
                    break
            if pick is None:
                pick = scored[0][1]
            ff.target = pick
            taken.append(pick)

    def _consume_pool(self, kind: PoiState) -> None:
        if kind == PoiState.VICTIM:
            self.victim_pool = max(0, self.victim_pool - 1)
        else:
            self.false_alarm_pool = max(0, self.false_alarm_pool - 1)

    def _draw_poi_kind(self) -> PoiState | None:
        total = self.victim_pool + self.false_alarm_pool
        if total <= 0:
            return None
        kind = PoiState.VICTIM if self.rng.integers(0, total) < self.victim_pool else PoiState.FALSE
        self._consume_pool(kind)
        return kind

    # ---- callbacks from Firefighter actions ----------------------------

    def _free_claim_at(self, pos: tuple[int, int]) -> None:
        """Suelta el reclamo de cualquier bombero cuya víctima esté (o
        estuviera) en `pos`, y limpia el dueño del POI que quede ahí."""
        for ff in self.firefighters:
            if ff.claimed_victim == pos:
                ff.claimed_victim = None
                ff.role = SUPPRESSION_ROLE
        poi = self.board.poi_at(pos)
        if poi is not None:
            poi.claim = None

    def _claim_spawned_pois(self) -> None:
        """Reponer POI puede dejar un marcador justo debajo de un bombero
        (y revelarlo ahí mismo). Ese es el único caso de claim-on-contact:
        el bombero que lo tiene encima se lo adjudica y pasa a evacuación."""
        for ff in self.firefighters:
            if ff.claimed_victim is not None:
                continue
            poi = self.board.poi_at(ff.pos)
            if (
                poi is None
                or poi.kind != PoiState.VICTIM
                or not poi.spawned_on_agent
                or poi.claim is not None
            ):
                continue
            ff.claimed_victim = poi.pos
            poi.claim = ff
            ff.role = EVACUATION_ROLE

    def rescue_victim(self, pos: tuple[int, int]) -> None:
        self.victims_rescued += 1
        self.board.remove_poi(pos)
        self._free_claim_at(pos)
        self._assign_victims()
        self._check_end_conditions()

    def identify_false_alarm(self, pos: tuple[int, int]) -> None:
        self.false_alarms_found += 1

    def register_damage(self, amount: int = 1) -> bool:
        self.damage_total += amount
        self._check_end_conditions()
        return self.status == "in_progress"

    def _lose_victim(self) -> None:
        self.victims_lost += 1
        self._assign_victims()
        self._check_end_conditions()

    # ---- turn cycle ---------------------------------------------------

    def step(self) -> None:
        if self.status != "in_progress":
            return

        self._assign_fire_targets()

        actor_idx = self.turn % len(self.firefighters)
        firefighter = self.firefighters[actor_idx]
        firefighter.take_turn()

        fire_event = None
        if self.status == "in_progress":
            fire_event = advance_fire(self.board, self.rng, on_damage=self.register_damage)
            self._apply_fire_event(fire_event)

        if self.status == "in_progress":
            firefighter_positions = {f.pos for f in self.firefighters}
            replenish_poi(self.board, self.rng, self._draw_poi_kind, firefighter_positions, POI_TARGET_ON_BOARD)
            self._claim_spawned_pois()
            self._assign_victims()

        self.turn += 1
        self.collector.collect(self, fire_event, actor_idx=actor_idx)

    def run(self, max_turns: int = 1000) -> str:
        if not self.collector.rows:
            self.collector.collect(self, None, initial=True)
        while self.status == "in_progress" and self.turn < max_turns:
            self.step()
        return self.status

    # ---- fire event consequences --------------------------------------

    def _apply_fire_event(self, event: FireEvent) -> None:
        for pos, kind in event.lost_pois:
            if self.status != "in_progress":
                break
            if kind == PoiState.VICTIM:
                self._free_claim_at(pos)
                self._lose_victim()

        for firefighter in self.firefighters:
            if self.status != "in_progress":
                break
            if firefighter.pos in event.ignited and not firefighter.knocked_down:
                self._knock_down(firefighter)

    def _knock_down(self, firefighter: Firefighter) -> None:
        if firefighter.role == EVACUATION_ROLE:
            # La víctima transportada muere: se cuenta como perdida y su
            # marcador se quita del tablero (nada de dobles conteos).
            self.board.remove_poi(firefighter.pos)
            self._free_claim_at(firefighter.pos)
            self._lose_victim()
        firefighter.knocked_down = True
        firefighter.pos = self._nearest_healing_zone(firefighter.pos)  # type: ignore[arg-type]

    def _nearest_healing_zone(self, pos: tuple[int, int]) -> tuple[int, int]:
        row, col = pos
        return min(
            HEALING_ZONES,
            key=lambda zone: abs(zone[0] - row) + abs(zone[1] - col),
        )

    def _check_end_conditions(self) -> None:
        if self.status != "in_progress":
            return
        if self.victims_rescued >= VICTIMS_TO_WIN:
            self.status = "won"
        elif self.victims_lost >= VICTIMS_LOST_TO_LOSE:
            self.status = "lost"
        elif self.damage_total >= COLLAPSE_DAMAGE:
            self.status = "lost"
