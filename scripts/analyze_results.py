#!/usr/bin/env python
"""Analiza los CSV que produce `run_experiments.py` (el resumen por
partida y, si se le pasa, el dataset completo por-turno) y explica QUE
esta pasando: por que se pierde por colapso, como crece el fuego, la
timeline de victimas, y que hacen los bomberos.

Uso:
    python scripts/analyze_results.py \
        [--summary data/batch_summary.csv] \
        [--timecsv data/batch_full.csv] \
        [--plots-dir data/plots]          # omitir con --no-plots
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLLAPSE_DAMAGE = 24


def _pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else 0.0


def _fmt_mean_tbl(df: pd.DataFrame, cols: list[str], by: str = "cause") -> pd.DataFrame:
    """Tabla de promedios por causa para varias columnas numericas."""
    return df.groupby(by)[cols].mean().round(2)


def report_overview(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("1. PANORAMA GENERAL")
    print("=" * 78)
    n = len(summary)
    causes = summary["cause"].value_counts()
    print(f"partidas analizadas: {n}")
    for c in ("won", "collapse", "victims_lost", "other"):
        cnt = int(causes.get(c, 0))
        print(f"  {c:<12} {cnt:4d}  ({_pct(cnt, n):.1f}%)")
    ties = int(
        ((summary["reached_damage_threshold"] == 1) & (summary["reached_victims_threshold"] == 1)).sum()
    )
    print(f"  cruces ambos umbrales el mismo turno (registrado como victims_lost): {ties}")
    print("\npromedios por desenlace:")
    print(
        _fmt_mean_tbl(summary, ["turns", "victims_rescued", "victims_lost", "damage_total"]).to_string()
    )


def report_collapse(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("2. COLAPSO: QUIEN GASTA EL PRESUPUESTO DE DANO")
    print("=" * 78)
    coll = summary[summary["cause"] == "collapse"]
    if len(coll) == 0:
        print("(ninguna partida perdida por colapso en esta muestra)")
        return
    print(f"partidas por colapso: {len(coll)}")
    print("\ndesglose promedio del dano acumulado:")

    def p90(s: pd.Series) -> float:
        return float(np.percentile(s, 90))

    print(
        coll[["damage_total", "chop_damage", "fire_damage"]].agg(
            ["mean", "median", p90]
        )
    )
    chop_frac = (coll["chop_damage"] / 24.0).mean()
    rel_share = (coll["chop_damage"] / (coll["chop_damage"] + coll["fire_damage"])).mean()
    print(
        f"> los hachazos propios cubren {100*chop_frac:.1f}% del umbral de colapso "
        f"(24) y son {100*rel_share:.1f}% del dano total acumulado"
    )
    print("\nque tan cerca estaba la victoria al colapsar (rescatadas ese turno):")
    hist = coll["victims_rescued"].value_counts().sort_index()
    print(
        f"  rescatadas >= 6: {(coll['victims_rescued']>=6).mean():.1%} | "
        f"=5: {((coll['victims_rescued']>=5)).mean():.1%} | "
        f"<=2: {(coll['victims_rescued']<=2).mean():.1%}"
    )
    print("\ncuando colapsa:")
    ct = pd.to_numeric(coll["collapse_turn"], errors="coerce")
    print(
        f"  turno: media {ct.mean():.0f} | p10 {ct.quantile(.10):.0f} "
        f"p50 {ct.quantile(.50):.0f} p90 {ct.quantile(.90):.0f}"
    )
    fl = pd.to_numeric(coll["first_loss_turn"], errors="coerce")
    con_vict_then = (~coll["first_loss_turn"].isna()).mean()
    mean_fl = fl.mean()
    print(f"  proporcion con perdidas de victima ANTES del colapso: {con_vict_then:.0%}"
          + (f" (primer perdida ~turno {mean_fl:.0f})" if mean_fl == mean_fl else ""))
    print("\nchop_damage por desenlace (promedio):")
    print(summary.groupby("cause")["chop_damage"].mean().round(2).to_string())
    print("\ncorrelacion de Pearson de perdida por la via del colapso:")
    corr = summary.assign(
        is_collapse=(summary["cause"] == "collapse").astype(int)
    )[["is_collapse", "chop_damage", "fire_damage", "n_chops", "victims_rescued", "turns"]].corr()[
        "is_collapse"
    ].round(3)
    print(corr.to_string())


def report_agent_behavior(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("3. QUE HACEN LOS BOMBEROS")
    print("=" * 78)
    act_cols = ["n_moves", "n_chops", "n_extinguish", "n_door", "n_reveal", "n_rescue"]
    print("acciones totales promedio por desenlace:")
    print(_fmt_mean_tbl(summary, act_cols).to_string())
    print("\nreparto final de roles promedio por desenlace (rescue/evac/suppression):")
    print(_fmt_mean_tbl(summary, ["end_rescue", "end_evacuation", "end_suppression"]).to_string())


def _add_full_features(full: pd.DataFrame) -> pd.DataFrame:
    """Anade a cada fila del dataset por-turno las columnas derivadas mas
    usadas a continuacion (fuego y abanico de dano)."""
    f = full.copy()
    fire_mask = f.filter(regex=r"^cell_\d+_\d+$").isin(["FIRE", "SMOKE"])
    f["fire_cells"] = fire_mask.sum(axis=1)
    f["fire_only_cells"] = f.filter(regex=r"^cell_\d+_\d+$").eq("FIRE").sum(axis=1)
    return f


def report_time_series(full: pd.DataFrame, summary: pd.DataFrame, plots_dir: str) -> None:
    print("\n" + "=" * 78)
    print("4. TIMELINE (dataset por-turno)")
    print("=" * 78)
    f = _add_full_features(full)
    f = f.merge(
        summary[["simulation_id", "cause"]], on="simulation_id", how="left"
    )

    per_run = f.groupby(["simulation_id", "cause"]).agg(
        turns=("turn", "count"),
        end_damage=("damage_total", "last"),
        end_rescued=("victims_rescued", "last"),
        end_lost=("victims_lost", "last"),
        peak_fire=("fire_cells", "max"),
        mean_fire=("fire_cells", "mean"),
        mean_fire_only=("fire_only_cells", "mean"),
        ignited=("fire_ignited_count", "sum"),
        firedmg=("fire_damage_added", "sum"),
    ).reset_index()

    agg = per_run.groupby("cause").agg(
        turns=("turns", "mean"),
        end_damage=("end_damage", "mean"),
        end_rescued=("end_rescued", "mean"),
        end_lost=("end_lost", "mean"),
        peak_fire=("peak_fire", "mean"),
        mean_fire=("mean_fire", "mean"),
        mean_fire_only=("mean_fire_only", "mean"),
        ignited=("ignited", "mean"),
        firedmg=("firedmg", "mean"),
    ).round(1)

    print("promedios por partida y desenlace (del dataset por-turno):")
    print(
        agg[["turns", "end_damage", "end_rescued", "end_lost",
             "peak_fire", "mean_fire", "mean_fire_only", "ignited", "firedmg"]]
        .rename(columns={
            "end_damage": "dano final", "end_rescued": "rescatadas",
            "end_lost": "perdidas", "peak_fire": "pico fuego+humo",
            "mean_fire": "media fuego+humo", "mean_fire_only": "media fuego puro",
            "ignited": "igniciones totales", "firedmg": "dmg-fuego total",
        })
        .to_string()
    )

    t = f.groupby(["turn", "cause"]).agg(
        damage=("damage_total", "mean"),
        rescued=("victims_rescued", "mean"),
        lost=("victims_lost", "mean"),
        fire=("fire_cells", "mean"),
    ).reset_index()

    if plots_dir:
        _plot_timeline(t, plots_dir)
        print(f"\n> graficos de timeline guardados en {plots_dir}")


def _plot_outcomes(summary: pd.DataFrame, plots_dir: str) -> None:
    vc = summary["cause"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(vc.values, labels=vc.index, autopct="%1.0f%%", startangle=90)
    ax.set_title("Desenlaces")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "outcomes_pie.png"), dpi=110)
    plt.close(fig)


def _plot_collapse_turns(summary: pd.DataFrame, plots_dir: str) -> None:
    coll = summary[summary["cause"] == "collapse"]
    if len(coll) == 0:
        return
    ct = pd.to_numeric(coll["collapse_turn"], errors="coerce").dropna()
    if len(ct) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ct, bins=min(40, int(ct.max()) // 2 + 1), color="#c0392b", alpha=0.85)
    ax.set_xlabel("turno de colapso")
    ax.set_ylabel("partidas")
    ax.set_title("Cuando colapsa el edificio")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "collapse_turns_hist.png"), dpi=110)
    plt.close(fig)


def _plot_chop_vs_rescued(summary: pd.DataFrame, plots_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"won": "#27ae60", "collapse": "#e74c3c", "victims_lost": "#f39c12"}
    for c in ("won", "collapse", "victims_lost"):
        sub = summary[summary["cause"] == c]
        if len(sub):
            ax.scatter(sub["chop_damage"], sub["victims_rescued"], s=14,
                       alpha=0.6, label=c, color=colors.get(c))
    ax.set_xlabel("dano por hachazos propios")
    ax.set_ylabel("victimas rescatadas")
    ax.set_title("Hachazos vs rescatadas")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "chop_vs_rescued.png"), dpi=110)
    plt.close(fig)


def _plot_action_counts(summary: pd.DataFrame, plots_dir: str) -> None:
    cols = ["n_moves", "n_chops", "n_extinguish", "n_door", "n_reveal", "n_rescue"]
    means = summary.groupby("cause")[cols].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    means.T.plot.bar(ax=ax)
    ax.set_ylabel("acciones promedio por partida")
    ax.set_title("Que hacen los bomberos (por desenlace)")
    ax.legend(title="desenlace")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "actions_bar.png"), dpi=110)
    plt.close(fig)


def _plot_timeline(t: pd.DataFrame, plots_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    lines = {"won": "-", "collapse": "--", "victims_lost": ":"}
    colors = {"won": "#27ae60", "collapse": "#e74c3c", "victims_lost": "#f39c12"}
    for (ax, ycol, title) in (
        (axes[0, 0], "damage", "Dano acumulado (m)"),
        (axes[0, 1], "fire", "Celdas fuego+humo (m)"),
        (axes[1, 0], "rescued", "Victimas rescatadas (m)"),
        (axes[1, 1], "lost", "Victimas perdidas (m)"),
    ):
        for cause, grp in t.groupby("cause"):
            ax.plot(grp["turn"], grp[ycol], linestyle=lines.get(cause, "-"),
                    color=colors.get(cause), label=cause)
        ax.set_title(title)
        ax.set_xlabel("turno")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "timeline.png"), dpi=110)
    plt.close(fig)


def report_takeaways(summary: pd.DataFrame) -> None:
    n = len(summary)
    won = (summary["cause"] == "won").mean()
    coll = summary[summary["cause"] == "collapse"]
    victims = summary[summary["cause"] == "victims_lost"]
    lines = []
    lines.append(f"Se ganan {100*won:.0f}% de las partidas y el colapso es la causa de derrota "
                 f"mas comun ({100*(len(coll)/n):.0f}%) vs victimas ({100*(len(victims)/n):.0f}%).")
    if len(coll):
        chop_pct = 100 * (coll["chop_damage"] / COLLAPSE_DAMAGE).mean()
        resc6 = 100 * (coll["victims_rescued"] >= 6).mean()
        lines.append(
            f"El colapso NO lo causan los propios hachazos: de media solo ~{chop_pct:.0f}% del "
            f"presupuesto de 24 es dano por cortar paredes; el resto llega de las EXPLOSIONES del "
            f"fuego golpeando las paredes."
        )
        lines.append(
            f"Al colapsar, la victoria suele estar lejos: solo {resc6:.0f}% de los colapsos tienen "
            f"6+ rescates hechos."
        )
    print("\n" + "=" * 78)
    print("RESUMEN")
    print("=" * 78)
    for l in lines:
        print(" *", l)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="data/batch_summary.csv")
    parser.add_argument("--timecsv", default=None, help="CSV completo por-turno (opcional)")
    parser.add_argument("--plots-dir", default="data/plots")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    for col in ("cause", "reached_damage_threshold", "reached_victims_threshold"):
        if col not in summary.columns:
            print(f"ERROR: falta la columna '{col}' en {args.summary} (es el CSV de resumen?)")
            sys.exit(1)

    plots_dir = None if args.no_plots else args.plots_dir
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)

    report_overview(summary)
    report_collapse(summary)
    report_agent_behavior(summary)

    if args.timecsv and os.path.exists(args.timecsv):
        print(f"(cargando dataset por-turno {args.timecsv}...)", file=sys.stderr)
        full = pd.read_csv(args.timecsv)
        report_time_series(full, summary, plots_dir)
    else:
        print("\n(no se dio --timecsv: se omiten los analisis de timeline)")

    if plots_dir:
        _plot_outcomes(summary, plots_dir)
        _plot_collapse_turns(summary, plots_dir)
        _plot_chop_vs_rescued(summary, plots_dir)
        _plot_action_counts(summary, plots_dir)
        print(f"> graficos guardados en {plots_dir}")

    report_takeaways(summary)


if __name__ == "__main__":
    main()