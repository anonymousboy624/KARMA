#!/usr/bin/env python
"""
experiments/latex_tables_cv.py
────────────────────────────────────────────────────────────────────────────
Generates one booktabs-style LaTeX table per (arch, metric) pair from
experiments/comparison_realdata_cv.py's *_cv{N}_results.json output files —
the same mean ± std numbers `print_summary_cv` prints to the terminal,
formatted for direct inclusion in a paper.

Table format matches:

    \\begin{table*}[t]
    \\centering
    \\caption{Mean $\\pm$ std <metric> (<ARCH>) over N folds. Higher is better ($\\uparrow$).}
    \\label{tab:<metric>_<arch>}
    \\setlength{\\tabcolsep}{4pt}
    \\small
    \\begin{tabular}{l|cccccc}
    \\toprule
    Explainer & Dataset1 & Dataset2 & ... \\\\
    \\midrule
    FO & 0.100 $\\pm$ 0.099 & ... \\\\
    ...
    TIMING & ... \\\\\\hline\\\\
    KARMA & \\textbf{0.784 $\\pm$ 0.067} & ... \\\\
    \\bottomrule
    \\end{tabular}
    \\end{table*}

Every method known to comparison_realdata_cv.py gets a row even when a
dataset has no data for it (shown as "--"), matching how the paper table
keeps FIT/WinIT rows visible while empty. The best value per column is
bolded — "best" means max for higher-is-better metrics (AUC, drop@25%) and
min for lower-is-better ones (complexity), rounded to 3 decimals so visual
ties (e.g. two methods printing "0.049") bold together. KARMA is
always the last row, set off by its own \\hline, since it's the reference
method every table is built to highlight.

Usage
-----
  python -m experiments.latex_tables_cv
  python -m experiments.latex_tables_cv --results_dir results/realdata_cv --n_folds 5
  python -m experiments.latex_tables_cv --archs tcn --metrics lag_drop25 lag_auc
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import json

DATASET_DISPLAY = {
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "exchange_rate": "ExRA",
    "beijing_pm25": "Bei",
    "web_traffic": "Web Traffic",
    "electricity": "Elec",
}
DATASET_ORDER = list(DATASET_DISPLAY.keys())

METHOD_DISPLAY = {
    "tsmule": "TS-MuLe",
    "shaptime": "ShapTime",
    "timeshap": "TimeSHAP",
    "timing": "TIMING",
    "karma": "KARMA",
}
# Baseline row order; KARMA is always appended last (own \hline), it's the
# reference method every table is built to highlight against the baselines.
METHOD_ORDER = [
    "tsmule",
    "shaptime",
    "timeshap",
    "extremalmask",
    "timing",
]

METRIC_DISPLAY = {
    "lag_auc": "Lag AUC",
    "lag_drop25": r"$\mathrm{AUC}_{lag}\text{@}25\%$",
    "complexity": "Complexity",
}
# True = higher is better (bold the max per column); False = bold the min.
METRIC_HIGHER_IS_BETTER = {
    "lag_auc": True,
    "lag_drop25": True,
    "complexity": False,
}

_KEY_RE = re.compile(r"^(gru|lstm|tcn)_([a-z0-9]+)_(.+)_mean$")


def load_results(results_dir: Path, n_folds: int) -> list[dict]:
    files = sorted(results_dir.glob(f"*_cv{n_folds}_results.json"))
    return [json.loads(f.read_text()) for f in files]


def discover_archs_methods_metrics(
    all_results: list[dict],
) -> tuple[list[str], list[str], list[str]]:
    archs, methods, metrics = set(), set(), set()
    for r in all_results:
        for k in r:
            m = _KEY_RE.match(k)
            if m:
                archs.add(m.group(1))
                methods.add(m.group(2))
                metrics.add(m.group(3))
    return sorted(archs), sorted(methods), sorted(metrics)


def _fmt_cell(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def build_table(
    all_results: list[dict],
    arch: str,
    metric: str,
    datasets_present: list[str],
    row_methods: list[str],
    n_folds: int,
) -> str:
    higher_better = METRIC_HIGHER_IS_BETTER.get(metric, True)
    metric_label = METRIC_DISPLAY.get(metric, metric)
    arrow = r"\uparrow" if higher_better else r"\downarrow"
    by_dataset = {r["dataset"]: r for r in all_results}

    # values[method][dataset] = (mean, std) or None
    values: dict[str, dict[str, tuple | None]] = {}
    for m in row_methods:
        values[m] = {}
        for ds in datasets_present:
            r = by_dataset.get(ds)
            mean = r.get(f"{arch}_{m}_{metric}_mean") if r else None
            std = r.get(f"{arch}_{m}_{metric}_std") if r else None
            values[m][ds] = (mean, std) if mean is not None else None

    # Best (rounded to display precision, so visual ties bold together).
    best: dict[str, float | None] = {}
    for ds in datasets_present:
        col_vals = [values[m][ds][0] for m in row_methods if values[m][ds] is not None]
        best[ds] = (max(col_vals) if higher_better else min(col_vals)) if col_vals else None

    def cell(m: str, ds: str) -> str:
        v = values[m][ds]
        if v is None:
            return "--"
        mean, std = v
        s = _fmt_cell(mean, std)
        if best[ds] is not None and round(mean, 3) == round(best[ds], 3):
            s = f"\\textbf{{{s}}}"
        return s

    col_spec = "l|" + "c" * len(datasets_present)
    header = "Explainer & " + " & ".join(
        DATASET_DISPLAY.get(d, d) for d in datasets_present
    ) + r" \\"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            rf"\caption{{Mean $\pm$ std {metric_label} ({arch.upper()}) over {n_folds} "
            rf"folds. {'Higher' if higher_better else 'Lower'} is better (${arrow}$).}}"
        ),
        rf"\label{{tab:{metric}_{arch}}}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    baseline_methods = [m for m in row_methods if m != "karma"]
    for i, m in enumerate(baseline_methods):
        row = METHOD_DISPLAY.get(m, m) + " & " + " & ".join(
            cell(m, ds) for ds in datasets_present
        )
        is_last_baseline = i == len(baseline_methods) - 1
        row += r" \\\hline\\" if (is_last_baseline and "karma" in row_methods) else r" \\"
        lines.append(row)

    if "karma" in row_methods:
        row = "KARMA & " + " & ".join(cell("karma", ds) for ds in datasets_present) + r" \\"
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description="Generate LaTeX tables from comparison_realdata_cv.py results"
    )
    p.add_argument("--results_dir", default="results/realdata_cv")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--out_dir", default=None, help="Defaults to <results_dir>/latex")
    p.add_argument(
        "--archs", nargs="+", default=None, help="Restrict to these archs (default: all found)"
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Restrict to these metrics (default: all found)",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "latex"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_results(results_dir, args.n_folds)
    if not all_results:
        print(f"No *_cv{args.n_folds}_results.json files found in {results_dir}")
        return

    found_archs, found_methods, found_metrics = discover_archs_methods_metrics(all_results)
    archs = args.archs or found_archs
    metrics = args.metrics or found_metrics

    datasets_present = [
        d for d in DATASET_ORDER if any(r["dataset"] == d for r in all_results)
    ]
    extra_datasets = sorted({r["dataset"] for r in all_results} - set(DATASET_ORDER))
    datasets_present += extra_datasets

    # Only currently-active methods (METHOD_DISPLAY) get a row. Older result
    # files may still carry keys for since-removed methods (e.g. contralsp,
    # timex) — silently drop those rather than surfacing them as ad-hoc rows.
    stale = sorted(set(found_methods) - set(METHOD_DISPLAY))
    if stale:
        print(f"Ignoring methods no longer in the pipeline: {stale}")
    row_methods = [m for m in METHOD_ORDER if m != "karma"]
    row_methods.append("karma")

    print(f"Datasets: {datasets_present}")
    print(f"Archs:    {archs}")
    print(f"Metrics:  {metrics}")
    print(f"Methods:  {row_methods}\n")

    all_tex = []
    for arch in archs:
        for metric in metrics:
            tex = build_table(
                all_results, arch, metric, datasets_present, row_methods, args.n_folds
            )
            fname = out_dir / f"{arch}_{metric}.tex"
            fname.write_text(tex + "\n")
            all_tex.append(tex)
            print(f"  wrote {fname}")

    combined = out_dir / f"all_tables_cv{args.n_folds}.tex"
    combined.write_text("\n\n".join(all_tex) + "\n")
    print(f"\nCombined: {combined}")


if __name__ == "__main__":
    main()
