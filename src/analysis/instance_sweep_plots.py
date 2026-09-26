"""Plots for instance-sweep results ({"<iteration>": (training_scores, eval_results)}).

Single run (box plot + profit/delay trends per model):
    python -m src.analysis.instance_sweep_plots --run runs/<run_id>

Cross-schedule-type comparison (random / bottleneck-"trap" / real-sample instance generation);
defaults to the historical 100-iteration pickles in data/experiments/:
    python -m src.analysis.instance_sweep_plots --compare [--random P] [--trap P] [--sample P]
"""

import argparse
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d

from src.analysis.results_io import load_results, sorted_iteration_keys
from src.experiments.experiment_setup import resolve_project_root
from src.utils import logger

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
PLOT_TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE, TICK_FONTSIZE, LEGEND_FONTSIZE = 17, 15, 13, 15


def plot_single_run(iter_viz: Dict[str, Any]) -> None:
    keys = sorted_iteration_keys(iter_viz)
    models = list(iter_viz[keys[0]][1].keys())
    iteration_indices = list(range(len(keys)))

    box_rewards = [[iter_viz[k][1][model][0] for k in keys] for model in models]
    plt.figure(figsize=(10, 5))
    plt.boxplot(box_rewards, tick_labels=models, showfliers=False)
    plt.title(f"Rewards across {len(keys)} iterations")
    plt.ylabel("Reward ($)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()

    colors = plt.cm.tab10(range(len(models)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    for ax, value_idx, ylabel, title in [
        (axes[0], 0, "Profit ($)", "Model Profit Across Iterations"),
        (axes[1], 1, "Delay (minutes)", "Model Delay Across Iterations"),
    ]:
        for idx, model in enumerate(models):
            values = [iter_viz[k][1][model][value_idx] for k in keys]
            ax.scatter(
                iteration_indices,
                values,
                alpha=0.4,
                s=100,
                color=colors[idx],
                label=model,
                edgecolors="black",
                linewidth=0.5,
            )
            if len(values) > 1:
                smoothed = uniform_filter1d(values, size=max(1, len(values) // 3))
                ax.plot(iteration_indices, smoothed, color=colors[idx], linewidth=2.5, alpha=0.8)
        ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()


def _shared_legend(fig, handles, labels, bottom_anchor: float) -> None:
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.525, bottom_anchor),
        ncol=max(1, len(labels) // 2),
        framealpha=0.95,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.8,
        borderpad=0.25,
        labelspacing=0.25,
    )


def plot_schedule_type_comparison(gathered_sets: list[tuple[str, Dict[str, Any]]]) -> None:
    last_dataset = gathered_sets[-1][1]
    models = list(last_dataset[sorted_iteration_keys(last_dataset)[0]][1].keys())
    colors = plt.cm.tab10(range(len(models)))
    model_colors = {name: colors[idx] for idx, name in enumerate(models)}

    summary_rows = []
    for schedule_name, dataset in gathered_sets:
        keys = sorted_iteration_keys(dataset)
        row: Dict[str, Any] = {"schedule": schedule_name}
        for model in models:
            row[model] = f"{np.mean([dataset[k][1][model][0] for k in keys]) / 1000:.1f}k"
        summary_rows.append(row)
    logger.info("\nAverage reward over iterations (per model, per schedule):")
    logger.info("\n" + pd.DataFrame(summary_rows).to_string(index=False))

    max_iterations = max(len(dataset) for _, dataset in gathered_sets)
    fig, axes = plt.subplots(len(gathered_sets), 1, figsize=(16, 10), sharex=True)
    for ax, (title, dataset) in zip(axes, gathered_sets):
        keys = sorted_iteration_keys(dataset)
        idx_range = list(range(len(keys)))
        for idx, model in enumerate(models):
            rewards = [dataset[k][1][model][0] for k in keys]
            ax.scatter(idx_range, rewards, alpha=0.35, s=85, color=colors[idx], edgecolors="black", linewidth=0.4)
            if len(rewards) > 1:
                ax.plot(
                    idx_range,
                    uniform_filter1d(rewards, size=max(1, len(rewards) // 3)),
                    color=colors[idx],
                    linewidth=2.8,
                    alpha=0.9,
                    linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
                    marker=MARKERS[idx % len(MARKERS)],
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    markevery=max(1, len(rewards) // 12),
                    label=model,
                )
        ax.set_title(f"{title}: Reward Across Iterations", fontsize=PLOT_TITLE_FONTSIZE, fontweight="bold")
        ax.set_ylabel("Reward ($)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(0, max_iterations - 1)
    axes[-1].set_xlabel("Iteration", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    _shared_legend(fig, *axes[0].get_legend_handles_labels(), bottom_anchor=0.0015)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.97))
    plt.show()

    fig, axes = plt.subplots(len(gathered_sets), 1, figsize=(16, 11), sharex=True)
    for ax, (schedule_name, dataset) in zip(axes, gathered_sets):
        keys = sorted_iteration_keys(dataset)
        boxplot = ax.boxplot(
            [[dataset[k][1][model][0] for k in keys] for model in models],
            patch_artist=True,
            widths=0.6,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for idx, box in enumerate(boxplot["boxes"]):
            box.set_facecolor(model_colors[models[idx]])
            box.set_alpha(0.55)
            box.set_edgecolor("black")
            box.set_linewidth(1.0)
        for line in boxplot["whiskers"] + boxplot["caps"]:
            line.set_color("black")
            line.set_linewidth(0.9)
        ax.set_title(
            f"{schedule_name}: Reward Distribution Across Iterations", fontsize=PLOT_TITLE_FONTSIZE, fontweight="bold"
        )
        ax.set_ylabel("Reward ($)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
        ax.tick_params(axis="y", which="major", labelsize=TICK_FONTSIZE)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[-1].set_xlabel("Model", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    axes[-1].set_xticks(np.arange(1, len(models) + 1))
    axes[-1].set_xticklabels(models, rotation=25, ha="right", fontsize=11)
    legend_handles = [Patch(facecolor=model_colors[m], edgecolor="black", alpha=0.55, label=m) for m in models]
    _shared_legend(fig, legend_handles, [h.get_label() for h in legend_handles], bottom_anchor=0.0015)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.97))
    plt.show()


def main() -> None:
    experiments_dir = resolve_project_root() / "data" / "experiments"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run folder or results pickle to plot on its own")
    parser.add_argument("--compare", action="store_true", help="plot the random/trap/sample comparison")
    parser.add_argument("--random", default=experiments_dir / "random_100_itr.pkl")
    parser.add_argument("--trap", default=experiments_dir / "trap_100_itr.pkl")
    parser.add_argument("--sample", default=experiments_dir / "sample_100_itr.pkl")
    args = parser.parse_args()

    if not args.run and not args.compare:
        parser.error("pass --run and/or --compare")
    if args.run:
        plot_single_run(load_results(args.run))
    if args.compare:
        plot_schedule_type_comparison(
            [
                ("Random Schedule", load_results(args.random)),
                ("Bottleneck Schedule", load_results(args.trap)),
                ("Sample Schedule", load_results(args.sample)),
            ]
        )


if __name__ == "__main__":
    main()
