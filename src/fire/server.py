"""Servidor Flask mínimo con un endpoint `/run-sim`.

Corre una partida completa y devuelve la serie de snapshots JSON
(documentada en `docs/json_schema.md`): turno 0 = estado inicial, turnos
1..N = estado tras la ronda. Sin parámetros usa el RNG por defecto de
mesa; con `?seed=N` reproduce exactamente la partida de `scripts/
run_experiments.py` con ese seed.
"""

from __future__ import annotations

from flask import Flask, jsonify, request

from fire.model import FirefighterModel

app = Flask(__name__)


def _make_model(seed: int | None) -> FirefighterModel:
    if seed is None:
        return FirefighterModel()
    import numpy as np

    from fire.board import PoiState

    init_rng = np.random.default_rng(1_000_000 + seed)

    def draw(self: FirefighterModel) -> PoiState | None:
        total = self.victim_pool + self.false_alarm_pool
        if total <= 0:
            return None
        kind = PoiState.VICTIM if init_rng.integers(0, total) < self.victim_pool else PoiState.FALSE
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


@app.get("/run-sim")
def run_sim():
    seed = request.args.get("seed", type=int, default=None)
    model = _make_model(seed)
    model.run()
    return jsonify(model.collector.snapshots)


if __name__ == "__main__":
    app.run(debug=True)
