from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from .board import Direction, PoiState

if TYPE_CHECKING:
    from .fire_rules import FireEvent
    from .model import FirefighterModel


def _fmt_cells(cells: set[tuple[int, int]]) -> str:
    """Formatea un set de posiciones como string compacto para CSV."""
    return ";".join(f"{r},{c}" for r, c in sorted(cells))


def _fmt_poi_losses(losses: list[tuple[tuple[int, int], PoiState]]) -> str:
    """Formatea la lista de POIs perdidos como string para CSV."""
    return ";".join(f"{r},{c}:{kind.name}" for (r, c), kind in losses)


class FireDataCollector:
    """Captura el estado completo del juego al final de cada turno para
    exportarlo a CSV y poder analizarlo después (por qué los agentes
    pierden, cómo se propaga el fuego, qué decisiones toman, etc.)."""

    def __init__(self, simulation_id: int = 0):
        self.simulation_id = simulation_id
        self.rows: list[dict] = []

    def collect(self, model: FirefighterModel, fire_event: FireEvent | None = None):
        """Toma un snapshot del estado completo del modelo y lo agrega
        como una fila más en `self.rows`."""
        row: dict = {}

        ff_id = model.turn % len(model.firefighters)
        ff = model.firefighters[ff_id]

        # -- metadata del turno --
        row["simulation_id"] = self.simulation_id
        row["turn"] = model.turn
        row["status"] = model.status
        row["firefighter_id"] = ff_id
        row["damage_total"] = model.damage_total
        row["victims_rescued"] = model.victims_rescued
        row["victims_lost"] = model.victims_lost
        row["false_alarms_found"] = model.false_alarms_found

        # -- evento de fuego --
        if fire_event is not None:
            row["fire_ignited_count"] = len(fire_event.ignited)
            row["fire_ignited_cells"] = _fmt_cells(fire_event.ignited)
            row["fire_lost_poi_count"] = len(fire_event.lost_pois)
            row["fire_lost_pois"] = _fmt_poi_losses(fire_event.lost_pois)
            row["fire_damage_added"] = fire_event.damage_added
        else:
            row["fire_ignited_count"] = 0
            row["fire_ignited_cells"] = ""
            row["fire_lost_poi_count"] = 0
            row["fire_lost_pois"] = ""
            row["fire_damage_added"] = 0

        # -- acciones del bombero que actuó este turno --
        row["ff_start_pos_r"] = ff.turn_start_pos[0]
        row["ff_start_pos_c"] = ff.turn_start_pos[1]
        row["actions"] = " ".join(ff.action_summary) if ff.action_summary else ""
        row["action_count"] = len(ff.action_summary)
        row["ap_spent"] = ff.ap_spent

        # -- estado de cada celda (8 × 10 = 80 columnas) --
        for r in range(model.board.height):
            for c in range(model.board.width):
                row[f"cell_{r}_{c}"] = model.board.get_state((r, c)).name

        # -- estado de cada pared (142 aristas canónicas) --
        # Horizontal: borde entre (r,c) y (r,c+1)
        for r in range(model.board.height):
            for c in range(model.board.width - 1):
                wall = model.board.get_wall((r, c), Direction.RIGHT)
                row[f"wall_h_{r}_{c}"] = wall.name if wall else "NONE"
        # Vertical: borde entre (r,c) y (r+1,c)
        for r in range(model.board.height - 1):
            for c in range(model.board.width):
                wall = model.board.get_wall((r, c), Direction.DOWN)
                row[f"wall_v_{r}_{c}"] = wall.name if wall else "NONE"

        # -- estado de cada bombero (6 agentes) --
        for i, f in enumerate(model.firefighters):
            fpos = f.pos if f.pos is not None else (-1, -1)
            fr, fc = fpos[0], fpos[1]  # type: ignore[union-attr]
            row[f"agent_{i}_pos_r"] = fr
            row[f"agent_{i}_pos_c"] = fc
            row[f"agent_{i}_ap"] = f.ap
            row[f"agent_{i}_saved_ap"] = f.saved_ap
            row[f"agent_{i}_carrying"] = int(f.carrying)
            row[f"agent_{i}_knocked_down"] = int(f.knocked_down)

        # -- POIs en el tablero (hasta 5 slots) --
        for i in range(5):
            if i < len(model.board.pois):
                poi = model.board.pois[i]
                row[f"poi_{i}_pos_r"] = poi.pos[0]
                row[f"poi_{i}_pos_c"] = poi.pos[1]
                row[f"poi_{i}_kind"] = poi.kind.name
                row[f"poi_{i}_revealed"] = int(poi.revealed)
            else:
                row[f"poi_{i}_pos_r"] = -1
                row[f"poi_{i}_pos_c"] = -1
                row[f"poi_{i}_kind"] = "NONE"
                row[f"poi_{i}_revealed"] = 0

        self.rows.append(row)

    def to_csv(self, path: str) -> None:
        """Escribe todas las filas recolectadas a un archivo CSV."""
        if not self.rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
