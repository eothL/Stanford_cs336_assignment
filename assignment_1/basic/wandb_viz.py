"""
Fetch W&B run data from a remote training server and visualize locally.

Usage:
    # Fetch all runs (or a specific one) and save locally
    python wandb_viz.py fetch
    python wandb_viz.py fetch --run-id <run_id>

    # Plot training curves from local data
    python wandb_viz.py plot
    python wandb_viz.py plot --run-id <run_id> --metrics loss,grad_norm

Requires: WANDB_ENTITY and WANDB_PROJECT env vars (or pass --entity / --project).
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "wandb_local"


# ── fetch ────────────────────────────────────────────────────────────────
def fetch_runs(entity: str, project: str, run_id: str | None = None):
    import wandb

    api = wandb.Api()
    DATA_DIR.mkdir(exist_ok=True)

    if run_id:
        runs = [api.run(f"{entity}/{project}/{run_id}")]
    else:
        runs = api.runs(f"{entity}/{project}")

    index = []
    for run in runs:
        # metadata
        meta = {
            "name": run.name,
            "id": run.id,
            "state": run.state,
            "url": run.url,
            "config": {
                k: v
                for k, v in dict(run.config).items()
                if not k.startswith("_")
            },
            "summary": {
                k: v
                for k, v in dict(run.summary).items()
                if not k.startswith("_")
            },
        }
        index.append(meta)

        # full history (step-by-step metrics)
        history = list(run.scan_history())
        run_dir = DATA_DIR / run.id
        run_dir.mkdir(exist_ok=True)

        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

        print(f"  saved {run.name} ({run.id}): {len(history)} steps")

    with open(DATA_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2, default=str)

    print(f"\n{len(index)} run(s) saved to {DATA_DIR}/")


# ── plot ─────────────────────────────────────────────────────────────────
def load_run_data(run_id: str) -> tuple[dict, list[dict]]:
    """Load metadata and history for a single run."""
    run_dir = DATA_DIR / run_id
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    with open(run_dir / "history.json") as f:
        history = json.load(f)
    return meta, history


def discover_metrics(history: list[dict]) -> list[str]:
    """Return all logged metric names (excluding internal wandb keys)."""
    keys = set()
    for row in history:
        keys.update(row.keys())
    return sorted(k for k in keys if not k.startswith("_"))


def plot_runs(run_ids: list[str], metrics: list[str] | None = None):
    """Plot training curves: one subplot per metric, all runs overlaid."""
    # load all runs first so we can auto-discover metrics if needed
    runs = []
    for run_id in run_ids:
        meta, history = load_run_data(run_id)
        runs.append((meta, history))

    # auto-discover metrics from the first run if not specified
    if metrics is None:
        metrics = discover_metrics(runs[0][1])

    # grid layout: aim for roughly square
    n = len(metrics)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[i // ncols][i % ncols]
        for meta, history in runs:
            # extract (step, value) pairs, skipping rows where metric is missing
            steps, vals = [], []
            for row in history:
                v = row.get(metric)
                if v is not None:
                    steps.append(row.get("_step", 0))
                    vals.append(v)
            if vals:
                ax.plot(steps, vals, label=meta["name"], alpha=0.8)
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.legend(fontsize="small")

    # hide unused subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.tight_layout()
    plt.show()
    


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="W&B local fetch & plot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # fetch
    fp = sub.add_parser("fetch", help="Download run data from W&B")
    fp.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    fp.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    fp.add_argument("--run-id", default=None, help="Fetch a single run")

    # plot
    pp = sub.add_parser("plot", help="Plot from locally saved data")
    pp.add_argument("--run-id", nargs="+", default=None,
                    help="Run ID(s) to plot. Default: all fetched runs.")
    pp.add_argument("--metrics", default=None,
                    help="Comma-separated metric names (default: auto-discover)")

    args = parser.parse_args()

    if args.cmd == "fetch":
        if not args.entity or not args.project:
            parser.error("Set WANDB_ENTITY and WANDB_PROJECT env vars, "
                         "or pass --entity and --project")
        fetch_runs(args.entity, args.project, args.run_id)

    elif args.cmd == "plot":
        if not DATA_DIR.exists():
            parser.error(f"No local data found. Run 'fetch' first. "
                         f"Expected: {DATA_DIR}/")

        # resolve run IDs
        if args.run_id:
            run_ids = args.run_id
        else:
            with open(DATA_DIR / "index.json") as f:
                run_ids = [r["id"] for r in json.load(f)]

        metrics = args.metrics.split(",") if args.metrics else None
        plot_runs(run_ids, metrics)


if __name__ == "__main__":
    main()
