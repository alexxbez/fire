#!/usr/bin/env python
"""Corre la simulacion N veces y exporta dos CSV:

  - `--out`: dataset completo (una fila por turno) con TODAS las columnas
    que ya produce el FireDataCollector (celdas, paredes, bomberos, POIs,
    acciones, fuego), con la columna `simulation_id` para distinguir cada
    partida.
  - `--summary-out`: un resumen compacto de una fila por partida, con el
    desenlace, la causa de la derrota, la timelina clave (primer perdida,
    turno de colapso) y el desglose del dano (propio hachazo vs fuego).

El inicial POI draw es determinista por seed (mesa crea un RNG sin semilla
por defecto), y el RNG de la simulacion se siembra con `--seed-base + i`,
asi cada corrida es reproducible.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import random
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/../src")

import numpy as np  # noqa: E402

from fire.agents import EVACUATION_ROLE, RESCUE_ROLE, SUPPRESSION_ROLE  # noqa: E402
from fire.board import PoiState  # noqa: E402
from fire.model import (  # noqa: E402
    COLLAPSE_DAMAGE,
    FirefighterModel,
    VICTIMS_LOST_TO_LOSE,
    VICTIMS_TO_WIN,
)


def make_model(seed: int) -> FirefighterModel:
    """Modelo con inicial POI draw determinista + RNG sembrado.

    FirefighterModel.__init__ saca los POIs iniciales del RNG sin semilla
    de mesa antes de que podamos sembrar `model.rng`; para que dos corridas
    con la misma seed empiecen identicas, se sobre-escribe temporalmente
    `_draw_poi_kind` con uno que usa un generador aparte fijado por seed."""
    init_rng = np.random.default_rng(1_000_000 + seed)

    def draw(self: FirefighterModel) -> PoiState | None:
        total = self.victim_pool + self.false_alarm_pool
        if total <= 0:
            return None
        x = int(init_rng.integers(0, total))
        kind = PoiState.VICTIM if x < self.victim_pool else PoiState.FALSE
        self._consume_pool(kind)
        return kind

    orig_draw = FirefighterModel._draw_poi_kind
    FirefighterModel._draw_poi_kind = draw
    try:
        model = FirefighterModel()
    finally:
        FirefighterModel._draw_poi_kind = orig_draw
    model.rng = np.random.default_rng(seed)
    return model


def classify(model: FirefighterModel) -> str:
    """Causa de la derrota tal como la registra el modelo: gana primero,
    luego perdida por victimas, luego colapso."""
    if model.status == "won":
        return "won"
    if model.victims_lost >= VICTIMS_LOST_TO_LOSE:
        return "victims_lost"
    if model.damage_total >= COLLAPSE_DAMAGE:
        return "collapse"
    return "other"


def summarize(model: FirefighterModel, sim_id: int) -> dict:
    """Deriva la fila de resumen de una partida ya terminada, a partir del
    estado final del modelo y de las filas por-turno ya recolectadas."""
    rows = model.collector.rows

    # Desglose del dano: el delta de damage_total por turno es
    # (hachazos hechos + dano por explosion de fuego). El hachazo propio
    # = max(0, delta - fire_damage_added).
    chops_dmg = 0
    fire_dmg = 0
    prev = 0
    for r in rows:
        dmg = int(r["damage_total"])
        fired = int(r["fire_damage_added"])
        fire_dmg += fired
        chops_dmg += max(0, (dmg - prev) - fired)
        prev = dmg

    actions = [tok.split(":")[0] for r in rows for tok in r["actions"].split() if tok]
    counts = Counter(actions)

    max_fire = 0
    fire_sum = 0
    for r in rows:
        nf = sum(1 for k, v in r.items() if k.startswith("cell_") and v in ("FIRE", "SMOKE"))
        max_fire = max(max_fire, nf)
        fire_sum += nf
    n_turns = len(rows) or 1

    first_loss = ""
    collapse_turn = ""
    for r in rows:
        if first_loss == "" and int(r["victims_lost"]) > 0:
            first_loss = int(r["turn"])
        if collapse_turn == "" and int(r["damage_total"]) >= COLLAPSE_DAMAGE:
            collapse_turn = int(r["turn"])

    role_mix = Counter(f.role for f in model.firefighters)

    return {
        "simulation_id": sim_id,
        "status": model.status,
        "cause": classify(model),
        "turns": model.turn,
        "victims_rescued": model.victims_rescued,
        "victims_lost": model.victims_lost,
        "damage_total": model.damage_total,
        "false_alarms_found": model.false_alarms_found,
        "reached_damage_threshold": int(model.damage_total >= COLLAPSE_DAMAGE),
        "reached_victims_threshold": int(model.victims_lost >= VICTIMS_LOST_TO_LOSE),
        "chop_damage": chops_dmg,
        "fire_damage": fire_dmg,
        "n_moves": counts.get("move", 0),
        "n_chops": counts.get("chop", 0),
        "n_extinguish": counts.get("extinguish", 0),
        "n_door": counts.get("door", 0),
        "n_reveal": counts.get("reveal", 0),
        "n_rescue": counts.get("rescue", 0),
        "peak_fire_cells": max_fire,
        "mean_fire_cells": round(fire_sum / n_turns, 2),
        "total_ignited": sum(int(r["fire_ignited_count"]) for r in rows),
        "first_loss_turn": first_loss,
        "collapse_turn": collapse_turn,
        "end_rescue": role_mix.get(RESCUE_ROLE, 0),
        "end_evacuation": role_mix.get(EVACUATION_ROLE, 0),
        "end_suppression": role_mix.get(SUPPRESSION_ROLE, 0),
        # contexto de los parametros con los que se corrio la fase
        "victims_to_win": VICTIMS_TO_WIN,
        "victims_lost_to_lose": VICTIMS_LOST_TO_LOSE,
        "collapse_damage": COLLAPSE_DAMAGE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=1000, help="numero de partidas (default 1000)")
    parser.add_argument("--out", default="data/batch_full.csv", help="CSV por-turno completo")
    parser.add_argument("--summary-out", default="data/batch_summary.csv", help="CSV resumen por partida")
    parser.add_argument("--seed-base", type=int, default=random.randint(1, 100), help="offset de semilla (default 1000)")
    parser.add_argument("--quiet", action="store_true", help="no imprimir progreso")
    args = parser.parse_args()

    t0 = time.time()
    summary_rows: list[dict] = []
    n_written = 0

    with open(args.out, "w", newline="", encoding="utf-8") as csvf:
        writer = None
        for i in range(args.sims):
            seed = args.seed_base + i
            model = make_model(seed)
            model.collector.simulation_id = i
            model.run()

            rows = model.collector.rows
            if writer is None and rows:
                writer = csv.DictWriter(csvf, fieldnames=list(rows[0].keys()))
                writer.writeheader()
            for r in rows:
                writer.writerow(r)
            summary_rows.append(summarize(model, i))
            n_written += 1

            if not args.quiet and (i % 100 == 99 or i == args.sims - 1):
                elapsed = time.time() - t0
                print(
                    f"[{i+1:4d}/{args.sims}] {elapsed:.1f}s | "
                    f"ej. turnos {model.turn:3d} rescued {model.victims_rescued} "
                    f"lost {model.victims_lost} dmg {model.damage_total}",
                    flush=True,
                )

    with open(args.summary_out, "w", newline="", encoding="utf-8") as f:
        swriter = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        swriter.writeheader()
        swriter.writerows(summary_rows)

    taken = time.time() - t0
    statuses = Counter(r["status"] for r in summary_rows)
    print(f"listo: {n_written} partidas en {taken:.1f}s")
    print(f"  por-turno   -> {args.out}")
    print(f"  resumen     -> {args.summary_out}")
    print(f"  desenlaces: {dict(statuses)}")


if __name__ == "__main__":
    main()
