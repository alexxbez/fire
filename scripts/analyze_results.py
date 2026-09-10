#!/usr/bin/env python
"""Analiza los CSVs que produce `run_experiments.py` (ambos agentes: smart
y random) y genera reportes + graficos comparativos.

Uso:
    python scripts/analyze_results.py [--data-dir DIR] [--plots-dir DIR] [--no-plots]
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

# ---- palette -----------------------------------------------------------

C = {
    "dark":   "#242424",
    "blue":   "#68abdf",
    "green":  "#99c47a",
    "yellow": "#ffc66d",
    "red":    "#ff6470",
    "white":  "#f2f2f2",
}

AGENT_COLORS = {"smart": C["blue"], "random": C["green"]}
CAUSE_COLORS = {"won": C["yellow"], "collapse": C["red"], "victims_lost": C["dark"], "other": C["white"]}


def _pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else 0.0


def _fmt_mean_tbl(df: pd.DataFrame, cols: list[str], by: str = "cause") -> pd.DataFrame:
    return df.groupby(by)[cols].mean().round(2)


# ---- data loading ------------------------------------------------------

def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    smart_sum = pd.read_csv(os.path.join(data_dir, "batch_smart_summary.csv"))
    rand_sum = pd.read_csv(os.path.join(data_dir, "batch_random_summary.csv"))
    summary = pd.concat([smart_sum, rand_sum], ignore_index=True)
    return smart_sum, rand_sum, summary


# ---- text reports ------------------------------------------------------

def report_overview(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("1. PANORAMA GENERAL")
    print("=" * 78)
    for agent in ("smart", "random"):
        sub = summary[summary["agent_type"] == agent]
        n = len(sub)
        causes = sub["cause"].value_counts()
        print(f"\n  [{agent}] {n} partidas")
        for c in ("won", "collapse", "victims_lost", "other"):
            cnt = int(causes.get(c, 0))
            print(f"    {c:<14} {cnt:4d}  ({_pct(cnt, n):.1f}%)")
        print(f"    promedios: turnos={sub['turns'].mean():.1f}  "
              f"rescatadas={sub['victims_rescued'].mean():.2f}  "
              f"perdidas={sub['victims_lost'].mean():.2f}  "
              f"dano={sub['damage_total'].mean():.1f}")


def report_comparative(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("2. COMPARATIVA DIRECTA")
    print("=" * 78)
    metrics = ["turns", "victims_rescued", "victims_lost", "damage_total",
               "chop_damage", "fire_damage", "peak_fire_cells"]
    tbl = summary.groupby("agent_type")[metrics].mean().round(2)
    print(tbl.to_string())
    print("\n ratio smart/random (valores >1 favorecen al smart):")
    if "smart" in tbl.index and "random" in tbl.index:
        for col in metrics:
            s, r = tbl.loc["smart", col], tbl.loc["random", col]
            ratio = s / r if r else float("inf")
            print(f"    {col:<22} {ratio:.2f}")


def report_collapse(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("3. COLAPSO POR AGENTE")
    print("=" * 78)
    for agent in ("smart", "random"):
        sub = summary[(summary["agent_type"] == agent) & (summary["cause"] == "collapse")]
        n_total = len(summary[summary["agent_type"] == agent])
        print(f"\n  [{agent}] {len(sub)} de {n_total} ({_pct(len(sub), n_total):.1f}%)")
        if len(sub) == 0:
            continue
        print(f"    dano promedio: total={sub['damage_total'].mean():.1f}  "
              f"chop={sub['chop_damage'].mean():.1f}  fire={sub['fire_damage'].mean():.1f}")
        ct = pd.to_numeric(sub["collapse_turn"], errors="coerce").dropna()
        if len(ct):
            print(f"    turno de colapso: media={ct.mean():.0f}  "
                  f"p10={ct.quantile(.10):.0f}  p50={ct.quantile(.50):.0f}  "
                  f"p90={ct.quantile(.90):.0f}")


def report_behavior(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("4. ACCIONES POR AGENTE")
    print("=" * 78)
    act_cols = ["n_moves", "n_chops", "n_extinguish", "n_door", "n_reveal", "n_rescue"]
    tbl = summary.groupby("agent_type")[act_cols].mean().round(1)
    print(tbl.to_string())


# ---- plots -------------------------------------------------------------

def _apply_style(ax: plt.Axes) -> None:
    """Apply consistent axis styling."""
    ax.tick_params(colors=C["dark"])
    for spine in ax.spines.values():
        spine.set_color(C["dark"])
    ax.xaxis.label.set_color(C["dark"])
    ax.yaxis.label.set_color(C["dark"])
    ax.title.set_color(C["dark"])


def plot_victims_rescued_hist(summary: pd.DataFrame, plots_dir: str) -> None:
    """Side-by-side histogram of victims rescued per simulation."""
    fig, ax = plt.subplots(figsize=(7, 4))
    max_rescued = int(summary["victims_rescued"].max()) + 1
    bins = np.arange(-0.5, max_rescued + 0.5, 1)

    for agent, color in AGENT_COLORS.items():
        sub = summary[summary["agent_type"] == agent]
        ax.hist(sub["victims_rescued"], bins=bins, alpha=0.65, color=color,
                label=agent, edgecolor=C["dark"], linewidth=0.5)

    ax.set_xlabel("Victimas rescatadas")
    ax.set_ylabel("Partidas")
    ax.set_title("Distribucion de victimas rescatadas por agente")
    ax.legend()
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "victims_rescued_hist.png"), dpi=150, facecolor=C["white"])
    plt.close(fig)


def plot_outcomes_comparison(summary: pd.DataFrame, plots_dir: str) -> None:
    """Grouped bar chart: outcome counts per agent."""
    agents = ["smart", "random"]
    causes = ["won", "collapse", "victims_lost"]
    x = np.arange(len(causes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, agent in enumerate(agents):
        sub = summary[summary["agent_type"] == agent]
        counts = [len(sub[sub["cause"] == c]) for c in causes]
        ax.bar(x + i * width, counts, width, label=agent,
               color=AGENT_COLORS[agent], edgecolor=C["dark"], linewidth=0.5)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Victoria", "Colapso", "Victimas perdidas"])
    ax.set_ylabel("Partidas")
    ax.set_title("Distribucion de desenlaces")
    ax.legend()
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "outcomes_comparison.png"), dpi=150, facecolor=C["white"])
    plt.close(fig)


def plot_actions_comparison(summary: pd.DataFrame, plots_dir: str) -> None:
    """Grouped bar chart: average action counts per agent."""
    act_cols = ["n_moves", "n_chops", "n_extinguish", "n_door", "n_reveal", "n_rescue"]
    labels = ["Movimientos", "Hachazos", "Extinciones", "Puertas", "Revelados", "Rescates"]
    agents = ["smart", "random"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, agent in enumerate(agents):
        sub = summary[summary["agent_type"] == agent]
        means = [sub[col].mean() for col in act_cols]
        ax.bar(x + i * width, means, width, label=agent,
               color=AGENT_COLORS[agent], edgecolor=C["dark"], linewidth=0.5)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Promedio por partida")
    ax.set_title("Acciones promedio por agente")
    ax.legend()
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "actions_comparison.png"), dpi=150, facecolor=C["white"])
    plt.close(fig)


def plot_damage_comparison(summary: pd.DataFrame, plots_dir: str) -> None:
    """Grouped bar chart: chop vs fire damage per agent."""
    agents = ["smart", "random"]
    categories = ["Hachazos", "Fuego"]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, agent in enumerate(agents):
        sub = summary[summary["agent_type"] == agent]
        vals = [sub["chop_damage"].mean(), sub["fire_damage"].mean()]
        ax.bar(x + i * width, vals, width, label=agent,
               color=AGENT_COLORS[agent], edgecolor=C["dark"], linewidth=0.5)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Dano promedio")
    ax.set_title("Desglose de dano acumulado")
    ax.legend()
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "damage_comparison.png"), dpi=150, facecolor=C["white"])
    plt.close(fig)


def plot_key_metrics_comparison(summary: pd.DataFrame, plots_dir: str) -> None:
    """Grouped bar chart: key metrics per agent."""
    metrics = ["turns", "victims_rescued", "peak_fire_cells"]
    labels = ["Turnos", "Rescatadas", "Pico fuego+humo"]
    agents = ["smart", "random"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, agent in enumerate(agents):
        sub = summary[summary["agent_type"] == agent]
        vals = [sub[col].mean() for col in metrics]
        ax.bar(x + i * width, vals, width, label=agent,
               color=AGENT_COLORS[agent], edgecolor=C["dark"], linewidth=0.5)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Valor promedio")
    ax.set_title("Metricas clave por agente")
    ax.legend()
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "key_metrics_comparison.png"), dpi=150, facecolor=C["white"])
    plt.close(fig)


# ---- per-agent plots (recolored) --------------------------------------

def plot_outcomes_pie(summary: pd.DataFrame, agent: str, plots_dir: str) -> None:
    sub = summary[summary["agent_type"] == agent]
    vc = sub["cause"].value_counts()
    colors = [CAUSE_COLORS.get(c, C["white"]) for c in vc.index]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(vc.values, labels=vc.index, autopct="%1.0f%%", startangle=90,
           colors=colors, textprops={"color": C["dark"]})
    ax.set_title(f"Desenlaces [{agent}]")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"outcomes_pie_{agent}.png"), dpi=150, facecolor=C["white"])
    plt.close(fig)


def plot_collapse_turns(summary: pd.DataFrame, agent: str, plots_dir: str) -> None:
    sub = summary[(summary["agent_type"] == agent) & (summary["cause"] == "collapse")]
    if len(sub) == 0:
        return
    ct = pd.to_numeric(sub["collapse_turn"], errors="coerce").dropna()
    if len(ct) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ct, bins=min(40, int(ct.max()) // 2 + 1), color=C["red"], alpha=0.85,
            edgecolor=C["dark"], linewidth=0.5)
    ax.set_xlabel("Turno de colapso")
    ax.set_ylabel("Partidas")
    ax.set_title(f"Cuando colapsa el edificio [{agent}]")
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"collapse_turns_{agent}.png"), dpi=150, facecolor=C["white"])
    plt.close(fig)


def plot_chop_vs_rescued(summary: pd.DataFrame, agent: str, plots_dir: str) -> None:
    sub = summary[summary["agent_type"] == agent]
    color = AGENT_COLORS[agent]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(sub["chop_damage"], sub["victims_rescued"], s=14, alpha=0.6,
               color=color, edgecolor=C["dark"], linewidth=0.3)
    ax.set_xlabel("Dano por hachazos propios")
    ax.set_ylabel("Victimas rescatadas")
    ax.set_title(f"Hachazos vs rescatadas [{agent}]")
    _apply_style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"chop_vs_rescued_{agent}.png"), dpi=150, facecolor=C["white"])
    plt.close(fig)


# ---- main --------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="directorio con los CSVs")
    parser.add_argument("--plots-dir", default="data/plots", help="directorio de salida para graficos")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    smart_sum, rand_sum, summary = load_data(args.data_dir)

    for col in ("cause", "agent_type", "reached_damage_threshold", "reached_victims_threshold"):
        if col not in summary.columns:
            print(f"ERROR: falta columna '{col}' en los CSVs")
            sys.exit(1)

    plots_dir = None if args.no_plots else args.plots_dir
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)

    report_overview(summary)
    report_comparative(summary)
    report_collapse(summary)
    report_behavior(summary)

    if plots_dir:
        plot_victims_rescued_hist(summary, plots_dir)
        plot_outcomes_comparison(summary, plots_dir)
        plot_actions_comparison(summary, plots_dir)
        plot_damage_comparison(summary, plots_dir)
        plot_key_metrics_comparison(summary, plots_dir)

        for agent in ("smart", "random"):
            plot_outcomes_pie(summary, agent, plots_dir)
            plot_collapse_turns(summary, agent, plots_dir)
            plot_chop_vs_rescued(summary, agent, plots_dir)

        print(f"\n> graficos guardados en {plots_dir}")

    print("\n" + "=" * 78)
    print("RESUMEN COMPARATIVO")
    print("=" * 78)
    for agent in ("smart", "random"):
        sub = summary[summary["agent_type"] == agent]
        win_rate = (sub["cause"] == "won").mean() * 100
        avg_rescued = sub["victims_rescued"].mean()
        print(f"  [{agent}] victoria: {win_rate:.0f}%  |  rescatadas media: {avg_rescued:.2f}")


if __name__ == "__main__":
    main()
