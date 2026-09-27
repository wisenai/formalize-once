"""Assemble summary.csv, report.md, and all figures from graded results."""
from __future__ import annotations

import numpy as np

from . import metrics, paths


def build_report() -> dict:
    paths.ensure_dirs()
    df = metrics.load_graded()
    acc = metrics.accuracy_table(df)
    slope = metrics.degradation_slope(acc)
    gap = metrics.robustness_gap(acc)
    fm = metrics.fail_mode_table(df)
    cm = metrics.compute_matched(df)

    acc.to_csv(paths.RESULTS / "summary.csv", index=False)
    slope.to_csv(paths.RESULTS / "degradation_slope.csv", index=False)
    gap.to_csv(paths.RESULTS / "robustness_gap.csv", index=False)
    cm.to_csv(paths.RESULTS / "compute_matched.csv", index=False)
    if not fm.empty:
        fm.to_csv(paths.RESULTS / "fail_modes.csv", index=False)

    figs = []
    for model_id in sorted(df["model_id"].unique()):
        for family in sorted(df["family"].unique()):
            out = paths.FIGURES / f"acc_vs_level_{family}_{_slug(model_id)}.png"
            if metrics.fig_accuracy_vs_level(acc, family, model_id, out):
                figs.append(out.name)
        out = paths.FIGURES / f"robustness_gap_{_slug(model_id)}.png"
        if metrics.fig_robustness_gap_heatmap(df, acc, model_id, out):
            figs.append(out.name)
        out = paths.FIGURES / f"fail_modes_{_slug(model_id)}.png"
        if metrics.fig_fail_modes(fm, model_id, out):
            figs.append(out.name)
    out = paths.FIGURES / "compute_matched.png"
    if metrics.fig_compute_matched(cm, out):
        figs.append(out.name)

    # capability-gating curve (v2 headline figure 5)
    from . import capability

    cap_info = capability.build_capability(df)
    figs.append("capability_gating.png")

    # paired McNemar: neurosymbolic vs plain / scaled
    mcn = []
    for model_id in df["model_id"].unique():
        for family in df["family"].unique():
            for b2 in ["plain_cot", "scaled"]:
                r = metrics.mcnemar_paired(df, model_id, family, "neurosymbolic", b2)
                if r:
                    mcn.append(r)

    _write_markdown(df, acc, slope, gap, fm, cm, mcn, figs)
    return {"n_records": len(df), "figures": figs, "mcnemar": mcn}


def _slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def _write_markdown(df, acc, slope, gap, fm, cm, mcn, figs):
    lines = ["# Symbol-Scramble — Results Report\n"]
    lines.append(f"- Graded records: **{len(df)}**")
    lines.append(f"- Models: {', '.join(sorted(df['model_id'].unique()))}")
    lines.append(f"- Branches: {', '.join(sorted(df['branch'].unique()))}")
    lines.append(f"- Levels: {', '.join(sorted(df['level'].unique(), key=lambda l: metrics.LEVEL_ORDER.get(l,9)))}")
    lines.append(f"- Total model calls cost tokens: {int(df['prompt_tokens'].sum()+df['completion_tokens'].sum()):,}\n")

    lines.append("## Accuracy by family / model / branch / level\n")
    for (fam, model), g in acc.groupby(["family", "model_id"]):
        lines.append(f"\n### {fam} — {model}\n")
        piv = g.pivot_table(index="branch", columns="level", values="accuracy")
        cols = [c for c in metrics.LEVEL_SEQ if c in piv.columns]
        piv = piv.reindex(columns=cols)
        lines.append(piv.round(3).to_markdown())

    lines.append("\n## Degradation slope (Δaccuracy per level; flatter = more robust)\n")
    lines.append(slope.round(4).to_markdown(index=False))

    lines.append("\n## Robustness gap (neuro-symbolic − plain / scaled), by level\n")
    lines.append(gap.round(3).to_markdown(index=False))

    if mcn:
        lines.append("\n## Paired McNemar (neuro-symbolic vs. baseline)\n")
        import pandas as pd
        lines.append(pd.DataFrame(mcn).round(4).to_markdown(index=False))

    if not fm.empty:
        lines.append("\n## Neuro-symbolic outcome attribution (counts)\n")
        lines.append(fm.to_markdown(index=False))

    lines.append("\n## Compute-matched (mean tokens vs accuracy)\n")
    lines.append(cm.round(3).to_markdown(index=False))

    cap_csv = paths.RESULTS / "capability_curve.csv"
    if cap_csv.exists():
        import pandas as pd

        cap = pd.read_csv(cap_csv)
        lines.append("\n## Capability-gating (per model)\n")
        lines.append(cap.round(3).to_markdown(index=False))

    lines.append("\n## Figures\n")
    for f in figs:
        lines.append(f"- `data/figures/{f}`")

    (paths.RESULTS / "report.md").write_text("\n".join(str(x) for x in lines))
