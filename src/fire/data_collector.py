from __future__ import annotations

import csv
import json
import re
from typing import TYPE_CHECKING

from .agents import EVACUATION_ROLE
from .board import Direction, PoiState

if TYPE_CHECKING:
    from .fire_rules import FireEvent
    from .model import FirefighterModel

_EXTINGUISH_RE = re.compile(r"^extinguish:(\w+)@\((\d+),\s*(\d+)\)$")


def _parse_actions(tokens: list[str]) -> list[dict]:
    """Convierte los tokens de acción (formato CSV, ej. "move:UP",
    "extinguish:FIRE@(2, 3)") en objetos de acción estructurados."""
    actions: list[dict] = []
    for tok in tokens:
        if tok == "rescue":
            actions.append({"action": "rescue"})
            continue
        kind, _, arg = tok.partition(":")
        if kind in ("move", "door", "chop"):
            actions.append({"action": kind, "direction": arg})
        elif kind == "reveal":
            actions.append({"action": "reveal", "kind": arg})
        elif kind == "extinguish":
            m = _EXTINGUISH_RE.match(tok)
            if m:
                state, r, c = m.groups()
                actions.append({"action": "extinguish", "state": state, "pos": [int(r), int(c)]})
            else:
                actions.append({"action": tok})
        else:
            actions.append({"action": tok})
    return actions


class FireDataCollector:
    """Captura el estado completo del juego al final de cada turno. Cada
    snapshot vive en `self.snapshots` como un objeto JSON anidado (la
    fuente de verdad) y de él se deriva la fila plana de `self.rows`
    para exportarlo a CSV. El turno 0 es el estado inicial del tablero
    ANTES de que actúe ningún bombero; los turnos siguientes (1, 2, ...)
    son el estado tras la acción del bombero y el avance del fuego."""

    def __init__(self, simulation_id: int = 0):
        self.simulation_id = simulation_id
        self.rows: list[dict] = []
        self.snapshots: list[dict] = []

    def collect(
        self,
        model: FirefighterModel,
        fire_event: FireEvent | None = None,
        *,
        initial: bool = False,
        actor_idx: int = -1,
    ):
        """Toma un snapshot del estado completo del modelo y lo agrega a
        `self.snapshots` (objeto JSON) y a `self.rows` (fila plana CSV).
        Con `initial=True` se captura el estado antes de que actúe
        cualquier agente (turno 0, sin `actor` ni evento de fuego)."""
        if initial:
            ff = None
            ff_id = -1
        else:
            ff_id = actor_idx if actor_idx >= 0 else model.turn % len(model.firefighters)
            ff = model.firefighters[ff_id]

        actor = None
        actor_tokens: list[str] = []
        if ff is not None:
            start = ff.turn_start_pos
            actor_tokens = list(ff.action_summary)
            actor = {
                "id": ff_id,
                "start_pos": [start[0], start[1]],
                "actions": _parse_actions(actor_tokens),
                "action_count": len(actor_tokens),
                "ap_spent": ff.ap_spent,
            }

        fe = None
        if fire_event is not None:
            fe = {
                "ignited_count": len(fire_event.ignited),
                "ignited_cells": [
                    [r, c] for r, c in sorted(fire_event.ignited)
                ],
                "lost_poi_count": len(fire_event.lost_pois),
                "lost_pois": [
                    {"pos": [pos[0], pos[1]], "kind": kind.name}
                    for pos, kind in fire_event.lost_pois
                ],
                "damage_added": fire_event.damage_added,
            }

        board = model.board

        def _wall_state(pos: tuple[int, int], direction: Direction) -> str:
            wall = board.get_wall(pos, direction)
            return wall.name if wall is not None else "NONE"

        walls_h = [
            [
                _wall_state((r, c), Direction.RIGHT)
                for c in range(board.width - 1)
            ]
            for r in range(board.height)
        ]
        walls_v = [
            [
                _wall_state((r, c), Direction.DOWN)
                for c in range(board.width)
            ]
            for r in range(board.height - 1)
        ]

        snapshot: dict = {
            "turn": model.turn,
            "status": model.status,
            "actor": actor,
            "score": {
                "damage_total": model.damage_total,
                "victims_rescued": model.victims_rescued,
                "victims_lost": model.victims_lost,
                "false_alarms_found": model.false_alarms_found,
            },
            "fire_event": fe,
            "board": {
                "cells": [
                    [board.get_state((r, c)).name for c in range(board.width)]
                    for r in range(board.height)
                ],
                "walls": {"horizontal": walls_h, "vertical": walls_v},
            },
            "firefighters": [
                {
                    "id": i,
                    "pos": list(f.pos) if f.pos is not None else [-1, -1],
                    "role": f.role,
                    "ap": f.ap,
                    "saved_ap": f.saved_ap,
                    "carrying": f.role == EVACUATION_ROLE,
                    "knocked_down": f.knocked_down,
                }
                for i, f in enumerate(model.firefighters)
            ],
            "pois": [
                {
                    "pos": [poi.pos[0], poi.pos[1]],
                    "kind": poi.kind.name,
                    "revealed": poi.revealed,
                }
                for poi in board.pois
            ],
        }

        self.snapshots.append(snapshot)
        self.rows.append(self._flatten(snapshot, actor_tokens))

    # ---- exportación ----------------------------------------------------

    def _flatten(self, snapshot: dict, actor_tokens: list[str]) -> dict:
        """Deriva la fila plana (una columna por cada campo del CSV actual)
        a partir del snapshot JSON."""
        row: dict = {}

        sc = snapshot["score"]
        act = snapshot["actor"]
        fe = snapshot["fire_event"]

        # -- metadata del turno --
        row["simulation_id"] = self.simulation_id
        row["turn"] = snapshot["turn"]
        row["status"] = snapshot["status"]
        row["firefighter_id"] = act["id"] if act is not None else -1
        row["damage_total"] = sc["damage_total"]
        row["victims_rescued"] = sc["victims_rescued"]
        row["victims_lost"] = sc["victims_lost"]
        row["false_alarms_found"] = sc["false_alarms_found"]

        # -- evento de fuego --
        if fe is not None:
            row["fire_ignited_count"] = fe["ignited_count"]
            row["fire_ignited_cells"] = ";".join(f"{r},{c}" for r, c in fe["ignited_cells"])
            row["fire_lost_poi_count"] = fe["lost_poi_count"]
            row["fire_lost_pois"] = ";".join(
                f"{p['pos'][0]},{p['pos'][1]}:{p['kind']}" for p in fe["lost_pois"]
            )
            row["fire_damage_added"] = fe["damage_added"]
        else:
            row["fire_ignited_count"] = 0
            row["fire_ignited_cells"] = ""
            row["fire_lost_poi_count"] = 0
            row["fire_lost_pois"] = ""
            row["fire_damage_added"] = 0

        # -- acciones del bombero que actuó este turno --
        if act is not None:
            row["ff_start_pos_r"] = act["start_pos"][0]
            row["ff_start_pos_c"] = act["start_pos"][1]
            row["actions"] = " ".join(actor_tokens)
            row["action_count"] = act["action_count"]
            row["ap_spent"] = act["ap_spent"]
        else:
            row["ff_start_pos_r"] = -1
            row["ff_start_pos_c"] = -1
            row["actions"] = ""
            row["action_count"] = 0
            row["ap_spent"] = 0

        # -- estado de cada celda (8 × 10 = 80 columnas) --
        cells = snapshot["board"]["cells"]
        for r, cell_row in enumerate(cells):
            for c, state in enumerate(cell_row):
                row[f"cell_{r}_{c}"] = state

        # -- estado de cada pared (142 aristas canónicas) --
        for r, wall_row in enumerate(snapshot["board"]["walls"]["horizontal"]):
            for c, state in enumerate(wall_row):
                row[f"wall_h_{r}_{c}"] = state
        for r, wall_row in enumerate(snapshot["board"]["walls"]["vertical"]):
            for c, state in enumerate(wall_row):
                row[f"wall_v_{r}_{c}"] = state

        # -- estado de cada bombero (6 agentes) --
        for agent in snapshot["firefighters"]:
            i = agent["id"]
            row[f"agent_{i}_pos_r"] = agent["pos"][0]
            row[f"agent_{i}_pos_c"] = agent["pos"][1]
            row[f"agent_{i}_role"] = agent["role"]
            row[f"agent_{i}_ap"] = agent["ap"]
            row[f"agent_{i}_saved_ap"] = agent["saved_ap"]
            row[f"agent_{i}_carrying"] = int(agent["carrying"])
            row[f"agent_{i}_knocked_down"] = int(agent["knocked_down"])

        # -- POIs en el tablero (hasta 5 slots) --
        pois = snapshot["pois"]
        for i in range(5):
            if i < len(pois):
                poi = pois[i]
                row[f"poi_{i}_pos_r"] = poi["pos"][0]
                row[f"poi_{i}_pos_c"] = poi["pos"][1]
                row[f"poi_{i}_kind"] = poi["kind"]
                row[f"poi_{i}_revealed"] = int(poi["revealed"])
            else:
                row[f"poi_{i}_pos_r"] = -1
                row[f"poi_{i}_pos_c"] = -1
                row[f"poi_{i}_kind"] = "NONE"
                row[f"poi_{i}_revealed"] = 0

        return row

    def to_csv(self, path: str) -> None:
        """Escribe todas las filas recolectadas a un archivo CSV."""
        if not self.rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)

    def to_json(self, path: str) -> None:
        """Escribe la serie completa de snapshots a un archivo JSON. El
        documento es un array: cada elemento es el estado de un turno
        (turno 0 = estado inicial antes de que actúe ningún agente)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshots, f, indent=2, ensure_ascii=False)
