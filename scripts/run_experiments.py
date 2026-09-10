#!/usr/bin/env python
"""Corre la simulacion N veces para CADA tipo de agente (smart y random) y
exporta 4 CSVs:

  - data/{agent}_full.csv:    dataset completo (una fila por turno)
  - data/{agent}_summary.csv: resumen compacto (una fila por partida)

Ambos agentes usan las mismas semillas para que las comparaciones sean
pareadas: mismas configuraciones de tablero/fuego/POI, distinto comportamiento.

Uso:
    python scripts/run_experiments.py [--sims N] [--seed-base N] [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import os
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
from fire.random_agent import RandomFirefighter  # noqa: E402

AGENTS = [
    ("smart", None),
    ("random", RandomFirefighter),
]


def make_model(seed: int, agent_cls=None) -> FirefighterModel:
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
        model = FirefighterModel(agent_cls=agent_cls)
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


def summarize(model: FirefighterModel, sim_id: int, agent_type: str) -> dict:
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
        "agent_type": agent_type,
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


def run_agent_batch(
    agent_label: str,
    agent_cls,
    sims: int,
    seed_base: int,
    data_dir: str,
    quiet: bool,
) -> None:
    """Run N simulations for a single agent type and write its CSVs."""
    t0 = time.time()
    summary_rows: list[dict] = []

    out_path = os.path.join(data_dir, f"batch_{agent_label}_full.csv")
    summary_path = os.path.join(data_dir, f"batch_{agent_label}_summary.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as csvf:
        writer = None
        for i in range(sims):
            seed = seed_base + i
            model = make_model(seed, agent_cls=agent_cls)
            model.collector.simulation_id = i
            model.run()

            rows = model.collector.rows
            if writer is None and rows:
                writer = csv.DictWriter(csvf, fieldnames=list(rows[0].keys()))
                writer.writeheader()
            for r in rows:
                writer.writerow(r)
            summary_rows.append(summarize(model, i, agent_label))
            n_written = i + 1

            if not quiet and (i % 100 == 99 or i == sims - 1):
                elapsed = time.time() - t0
                print(
                    f"  [{agent_label}] [{i+1:4d}/{sims}] {elapsed:.1f}s | "
                    f"turnos {model.turn:3d} rescued {model.victims_rescued} "
                    f"lost {model.victims_lost} dmg {model.damage_total}",
                    flush=True,
                )

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        swriter = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        swriter.writeheader()
        swriter.writerows(summary_rows)

    taken = time.time() - t0
    statuses = Counter(r["status"] for r in summary_rows)
    print(f"  [{agent_label}] listo: {n_written} partidas en {taken:.1f}s")
    print(f"    por-turno   -> {out_path}")
    print(f"    resumen     -> {summary_path}")
    print(f"    desenlaces: {dict(statuses)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=1000, help="numero de partidas por agente (default 1000)")
    parser.add_argument("--data-dir", default="data", help="directorio de salida (default data/)")
    parser.add_argument("--seed-base", type=int, default=random.randint(1, 100), help="offset de semilla")
    parser.add_argument("--quiet", action="store_true", help="no imprimir progreso")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    t0 = time.time()
    for agent_label, agent_cls in AGENTS:
        if not args.quiet:
            print(f"\n--- {agent_label} ---", flush=True)
        run_agent_batch(agent_label, agent_cls, args.sims, args.seed_base, args.data_dir, args.quiet)

    taken = time.time() - t0
    print(f"\ntotal: {args.sims} partidas x {len(AGENTS)} agentes en {taken:.1f}s")


if __name__ == "__main__":
    main()
