from __future__ import annotations

import random as _random

from .agents import (
    EVACUATION_ROLE,
    RESCUE_ROLE,
    SUPPRESSION_ROLE,
    STARTING_AP,
    MAX_SAVED_AP,
    _MAX_ACTIONS_PER_TURN,
    Firefighter,
)
from .board import CellState, Direction, PoiState, WallState


class RandomFirefighter(Firefighter):
    """Un bombero con estrategia aleatoria: apaga fuego/humo en su celda,
    revela POI, abre puertas bloqueantes y se mueve al azar. Si encuentra
    una víctima, la transporta hasta el exterior."""

    def _act(self) -> None:
        for _ in range(_MAX_ACTIONS_PER_TURN):
            if self.ap <= 0 or self.model.status != "in_progress":
                return

            # 1) Siempre apagar fuego/humo en la propia celda
            if self.extinguish(None):
                continue

            # 2) Revelar POI bajo nuestros pies
            poi_here = self.board.poi_at(self.pos)
            if poi_here is not None and not poi_here.revealed:
                kind = self.board.reveal_poi(self.pos)
                self.action_summary.append(f"reveal:{kind.name}")
                if kind == PoiState.FALSE:
                    self.model.identify_false_alarm(self.pos)
                continue

            self._refresh_role()

            if self.role == EVACUATION_ROLE:
                if not self._random_step_toward_exterior():
                    return
            elif self.role == RESCUE_ROLE:
                if not self._random_step_toward_victim():
                    return
            else:
                if not self._random_step_suppress():
                    return

    # ---- random movement helpers ------------------------------------

    def _random_step_toward_exterior(self) -> bool:
        """Mueve un paso al azar hacia una dirección que nos acerque a una
        celda exterior. Si la dirección está bloqueada por fuego/humo, la
        apaga; si hay puerta cerrada, la abre."""
        exterior = self._exterior_cells()
        if not exterior:
            return False

        best_dir = None
        best_dist = None
        my_dist = min(abs(self.pos[0] - e[0]) + abs(self.pos[1] - e[1]) for e in exterior)

        for d in Direction:
            n = self._neighbor(d)
            if n is None:
                continue
            if self.board.has_poi(n) and n != self.pos:
                continue
            new_dist = min(abs(n[0] - e[0]) + abs(n[1] - e[1]) for e in exterior)
            if best_dist is None or new_dist < best_dist:
                best_dist = new_dist
                best_dir = d

        if best_dir is None:
            return False

        return self._try_action_in_direction(best_dir, carry=True)

    def _random_step_toward_victim(self) -> bool:
        """Mueve un paso al azar hacia la víctima reclamada."""
        victim = self.claimed_victim
        if victim is None:
            return False

        best_dir = None
        best_dist = None

        for d in Direction:
            n = self._neighbor(d)
            if n is None:
                continue
            new_dist = abs(n[0] - victim[0]) + abs(n[1] - victim[1])
            if best_dist is None or new_dist < best_dist:
                best_dist = new_dist
                best_dir = d

        if best_dir is None:
            return False

        return self._try_action_in_direction(best_dir, carry=False)

    def _random_step_suppress(self) -> bool:
        """Mueve un paso al azar en una dirección válida."""
        directions = list(Direction)
        _random.shuffle(directions)

        for d in directions:
            if self._try_action_in_direction(d, carry=False):
                return True
        return False

    def _try_action_in_direction(self, direction: Direction, carry: bool = False) -> bool:
        """Intenta moverse en una dirección dada. Si hay puerta cerrada, la
        abre; si hay fuego/humo, lo apaga; si es transitable, se mueve."""
        n = self._neighbor(direction)
        if n is None:
            return False

        # Abrir puerta cerrada antes de mirar la celda de al lado: a través
        # de una puerta cerrada no se puede apagar ni moverse.
        wall = self.board.get_wall(self.pos, direction)
        if wall == WallState.DOOR_CLOSE:
            if self.open_close_door(direction):
                return True
            return False

        # Apagar fuego/humo en la celda destino
        state = self.board.get_state(n)
        if state in (CellState.FIRE, CellState.SMOKE):
            if self.extinguish(direction):
                return True
            return False

        # Mover si es transitable
        if self.move(direction, carry=carry):
            return True

        return False
