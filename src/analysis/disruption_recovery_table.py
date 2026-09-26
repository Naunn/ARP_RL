"""Recovery-profit table for disruption-recovery results (see disruption_training_experiment.py).

    python -m src.analysis.disruption_recovery_table                         # historical Random/Trap/Sample pickles
    python -m src.analysis.disruption_recovery_table Trap=runs/<run_id> ...  # any LABEL=PATH (run folder or .pkl)

Columns:
  Initial Schedule : reward on the initial schedule (start -> after all retraining, with % change)
  Pre-Disrupt Avg  : average profit on the disrupted schedule BEFORE retraining
  Post-Retrain Avg : average profit on the disrupted schedule AFTER retraining
  Final Eval Avg   : average profit in the final re-evaluation on all disrupted schedules
  Avg Recovery %   : average % of lost profit recovered through retraining
"""

import argparse
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.analysis.results_io import load_results, sorted_iteration_keys
from src.experiments.experiment_setup import resolve_project_root
from src.utils import logger


def recovery_rows(schedule_type: str, iter_viz: Dict[str, Any]) -> list[dict]:
    run_keys = sorted_iteration_keys(iter_viz)
    if not run_keys:
        raise ValueError(f"No runs in {schedule_type} results.")
    disruption_keys = sorted(set.intersection(*(set(iter_viz[k][2].keys()) for k in run_keys)), key=int)
    if not disruption_keys:
        raise ValueError(f"No common disruption keys in {schedule_type} results.")

    # Every evaluated strategy except the random baseline, in the order the results recorded them.
    models = [m for m in iter_viz[run_keys[0]][1] if m != "Random Baseline"]

    rows = []
    for model in models:
        initial_start = float(np.mean([iter_viz[r][1][model][0] for r in run_keys]))
        initial_after = float(np.mean([iter_viz[r][2][d][3][model][0] for r in run_keys for d in disruption_keys]))
        initial_change_pct = (initial_after - initial_start) / initial_start * 100 if initial_start != 0 else 0
        pre = [iter_viz[r][2][d][1][model][0] for r in run_keys for d in disruption_keys]
        post = [iter_viz[r][2][d][2][model][0] for r in run_keys for d in disruption_keys]
        final = [iter_viz[r][3][d][model][0] for r in run_keys for d in disruption_keys]
        recovery_pcts = [
            (p_post - p_pre) / abs(initial_start - p_pre) * 100 if initial_start != p_pre else 0
            for p_pre, p_post in zip(pre, post)
        ]
        rows.append(
            {
                "Schedule Type": schedule_type,
                "Model": model,
                "Initial Schedule": f"${initial_start:.0f} -> ${initial_after:.0f} ({initial_change_pct:+.1f}%)",
                "Pre-Disrupt Avg": f"${np.mean(pre):.0f}",
                "Post-Retrain Avg": f"${np.mean(post):.0f}",
                "Final Eval Avg": f"${np.mean(final):.0f}",
                "Avg Recovery %": f"{np.mean(recovery_pcts):.1f}%",
            }
        )
    return rows


def main() -> None:
    experiments_dir = resolve_project_root() / "data" / "experiments"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "sources",
        nargs="*",
        default=[
            f"Random={experiments_dir / 'random_disruption_10_itr.pkl'}",
            f"Trap={experiments_dir / 'trap_disruption_10_itr.pkl'}",
            f"Sample={experiments_dir / 'sample_disruption_10_itr.pkl'}",
        ],
        help="LABEL=PATH pairs, PATH being a run folder or a results pickle",
    )
    args = parser.parse_args()

    rows = []
    for source in args.sources:
        label, _, path = source.partition("=")
        logger.info(f"Loading disruption recovery data from {path}")
        rows.extend(recovery_rows(label, load_results(path)))

    logger.info("\n" + "=" * 80 + "\nRECOVERY PROFIT ANALYSIS TABLE\n" + "=" * 80)
    logger.info("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
