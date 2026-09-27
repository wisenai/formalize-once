"""Capability-gating analysis (IMPLEMENTATION_SPEC v2 §11, figure 5).

The paper's central claim is that the neuro-symbolic advantage is gated by
formalization capability. We make that a curve: for each model, x = its own L0
plain-CoT accuracy (an internal capability proxy), y = (a) the neuro-symbolic
robustness gap over plain-CoT and (b) the neuro-symbolic formalization-success
rate. Report Spearman rho so the trend is quantified, not asserted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics, paths


def capability_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: capability proxy, robustness gap, formalization success."""
    rows = []
    for model_id, g in df.groupby("model_id"):
        eq = g[g["family"] == "equation"]
        if eq.empty:
            eq = g
        # capability proxy: L0 plain-CoT accuracy (fallback: all-level plain accuracy)
        l0p = eq[(eq["branch"] == "plain_cot") & (eq["level"] == "L0")]
        cap = float(l0p["correct"].mean()) if len(l0p) else float(
            eq[eq["branch"] == "plain_cot"]["correct"].mean()
        )
        # robustness gap = mean(neurosym) - mean(plain) over all levels
        ns = eq[eq["branch"] == "neurosymbolic"]
        pl = eq[eq["branch"] == "plain_cot"]
        gap = (float(ns["correct"].mean()) - float(pl["correct"].mean())
               if len(ns) and len(pl) else np.nan)
        ob = eq[eq["branch"] == "open_book"]
        gap_ob = (float(ns["correct"].mean()) - float(ob["correct"].mean())
                  if len(ns) and len(ob) else np.nan)
        # formalization-success rate: of neuro-symbolic outputs, fraction whose program
        # is right (correct, or "arithmetic" = right program wrong number) i.e. NOT a
        # formalization/execution failure.
        if len(ns):
            fm = ns["fail_mode"].fillna("none")
            formal_ok = fm.isin(["none", "arithmetic"]).mean()
        else:
            formal_ok = np.nan
        rows.append({
            "model_id": model_id, "capability_L0": cap,
            "robustness_gap": gap, "gap_vs_openbook": gap_ob,
            "formalization_success": float(formal_ok),
            "n": len(eq),
        })
    return pd.DataFrame(rows).sort_values("capability_L0")


def spearman(x, y):
    from scipy.stats import spearmanr

    m = ~(pd.isna(x) | pd.isna(y))
    if m.sum() < 3:
        return (np.nan, np.nan)
    r = spearmanr(np.asarray(x)[m], np.asarray(y)[m])
    return (float(r.statistic), float(r.pvalue))


def fig_capability_gating(tbl: pd.DataFrame, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = tbl["capability_L0"].to_numpy()
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.plot(x, tbl["robustness_gap"], "-o", color="#d62728",
            label="robustness gap (neuro-sym − plain)")
    ax.plot(x, tbl["formalization_success"], "-s", color="#1f77b4",
            label="formalization-success rate")
    for _, r in tbl.iterrows():
        ax.annotate(r["model_id"], (r["capability_L0"], r["robustness_gap"]),
                    fontsize=6, xytext=(3, -8), textcoords="offset points")
    rho, p = spearman(tbl["capability_L0"], tbl["robustness_gap"])
    ax.set_xlabel("model capability (L0 plain-CoT accuracy)")
    ax.set_ylabel("robustness gap / formalization success")
    title = "Capability-gating: code advantage grows with capability"
    if not np.isnan(rho):
        title += f"\nSpearman ρ(cap, gap) = {rho:.2f} (p = {p:.3f})"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_capability(df: pd.DataFrame) -> dict:
    tbl = capability_table(df)
    tbl.to_csv(paths.RESULTS / "capability_curve.csv", index=False)
    fig_capability_gating(tbl, paths.FIGURES / "capability_gating.png")
    rho, p = spearman(tbl["capability_L0"], tbl["robustness_gap"])
    return {"rho": rho, "p": p, "n_models": len(tbl)}
