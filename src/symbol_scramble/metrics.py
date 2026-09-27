"""Metrics + figures (SPEC §11): accuracy, robustness gap, slope, CIs, fail-modes."""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from . import paths

# ordinal position of each level for the degradation slope
LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "Lu": 3, "L3": 4, "L4": 5, "Flip": 5}
LEVEL_SEQ = ["L0", "L1", "L2", "Lu", "L3", "L4", "Flip"]
BRANCH_LABEL = {"plain_cot": "plain-CoT", "scaled": "scaled",
                "neurosymbolic": "neuro-symbolic", "open_book": "open-book"}
BRANCH_COLOR = {"plain_cot": "#1f77b4", "scaled": "#2ca02c",
                "neurosymbolic": "#d62728", "open_book": "#9467bd"}


def load_graded() -> pd.DataFrame:
    rows = []
    with open(paths.RESULTS / "graded.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["correct"] = df["correct"].astype(bool)
    return df


def _boot_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = vals.mean()
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boot = vals[idx].mean(axis=1)
    return (float(means), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (family, model, branch, level) with bootstrap 95% CI."""
    recs = []
    for (fam, model, branch, level), g in df.groupby(["family", "model_id", "branch", "level"]):
        vals = g["correct"].to_numpy().astype(float)
        acc, lo, hi = _boot_ci(vals)
        recs.append({
            "family": fam, "model_id": model, "branch": branch, "level": level,
            "n": len(vals), "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
            "level_ord": LEVEL_ORDER.get(level, 99),
        })
    return pd.DataFrame(recs).sort_values(["family", "model_id", "branch", "level_ord"])


def degradation_slope(acc_tbl: pd.DataFrame) -> pd.DataFrame:
    """OLS slope of accuracy vs ordinal level (L0..L4), per (family, model, branch)."""
    recs = []
    core = acc_tbl[acc_tbl["level"].isin(["L0", "L1", "L2", "Lu", "L3", "L4"])]
    for (fam, model, branch), g in core.groupby(["family", "model_id", "branch"]):
        g = g.sort_values("level_ord")
        if len(g) < 2:
            continue
        x = g["level_ord"].to_numpy(float)
        y = g["accuracy"].to_numpy(float)
        slope = np.polyfit(x, y, 1)[0]
        recs.append({"family": fam, "model_id": model, "branch": branch,
                     "slope": float(slope), "acc_L0": float(y[0]), "n_levels": len(g)})
    return pd.DataFrame(recs)


def robustness_gap(acc_tbl: pd.DataFrame) -> pd.DataFrame:
    """acc(neurosymbolic) - acc(plain_cot) and - acc(scaled), per (family,model,level)."""
    recs = []
    for (fam, model, level), g in acc_tbl.groupby(["family", "model_id", "level"]):
        by = {r["branch"]: r["accuracy"] for _, r in g.iterrows()}
        ns = by.get("neurosymbolic")
        if ns is None:
            continue
        recs.append({
            "family": fam, "model_id": model, "level": level,
            "level_ord": LEVEL_ORDER.get(level, 99),
            "gap_vs_plain": (ns - by["plain_cot"]) if "plain_cot" in by else np.nan,
            "gap_vs_scaled": (ns - by["scaled"]) if "scaled" in by else np.nan,
        })
    return pd.DataFrame(recs).sort_values(["family", "model_id", "level_ord"])


def mcnemar_paired(df: pd.DataFrame, model_id: str, family: str,
                   b1: str, b2: str) -> Optional[dict]:
    """Paired McNemar test on identical variant_ids between two branches."""
    from scipy.stats import binomtest

    sub = df[(df["model_id"] == model_id) & (df["family"] == family)]
    p = sub[sub["branch"] == b1].set_index("variant_id")["correct"]
    q = sub[sub["branch"] == b2].set_index("variant_id")["correct"]
    common = p.index.intersection(q.index)
    if len(common) == 0:
        return None
    p, q = p.loc[common], q.loc[common]
    b = int(((p) & (~q)).sum())   # b1 right, b2 wrong
    c = int(((~p) & (q)).sum())   # b2 right, b1 wrong
    n = b + c
    pval = binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    return {"model_id": model_id, "family": family, "b1": b1, "b2": b2,
            "n_pairs": len(common), "b1_only": b, "b2_only": c, "p_value": float(pval)}


def fail_mode_table(df: pd.DataFrame) -> pd.DataFrame:
    ns = df[df["branch"] == "neurosymbolic"].copy()
    if ns.empty:
        return pd.DataFrame()
    recs = []
    for (model, level), g in ns.groupby(["model_id", "level"]):
        counts = g["fail_mode"].value_counts(dropna=False).to_dict()
        rec = {"model_id": model, "level": level, "level_ord": LEVEL_ORDER.get(level, 99),
               "n": len(g)}
        for fm in ["none", "formalization", "execution", "arithmetic"]:
            rec[fm] = int(counts.get(fm, 0))
        recs.append(rec)
    return pd.DataFrame(recs).sort_values(["model_id", "level_ord"])


def compute_matched(df: pd.DataFrame) -> pd.DataFrame:
    """Mean total tokens vs accuracy per (family, model, branch)."""
    df = df.copy()
    df["tokens"] = df["prompt_tokens"] + df["completion_tokens"]
    recs = []
    for (fam, model, branch), g in df.groupby(["family", "model_id", "branch"]):
        recs.append({"family": fam, "model_id": model, "branch": branch,
                     "mean_tokens": float(g["tokens"].mean()),
                     "accuracy": float(g["correct"].mean()), "n": len(g)})
    return pd.DataFrame(recs)


# ---- figures -----------------------------------------------------------------


def _ensure_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_accuracy_vs_level(acc_tbl: pd.DataFrame, family: str, model_id: str, out):
    plt = _ensure_mpl()
    seq = [l for l in LEVEL_SEQ if l != ("Flip" if family == "equation" else "L4")]
    sub = acc_tbl[(acc_tbl["family"] == family) & (acc_tbl["model_id"] == model_id)]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = BRANCH_COLOR
    for branch in ["plain_cot", "scaled", "neurosymbolic", "open_book"]:
        b = sub[sub["branch"] == branch].set_index("level").reindex(seq).dropna(subset=["accuracy"])
        if b.empty:
            continue
        xs = [seq.index(l) for l in b.index]
        ax.plot(xs, b["accuracy"], "-o", color=colors[branch], label=BRANCH_LABEL[branch])
        ax.fill_between(xs, b["ci_lo"], b["ci_hi"], color=colors[branch], alpha=0.15)
    ax.set_xticks(range(len(seq)))
    ax.set_xticklabels(seq)
    ax.set_xlabel("perturbation level")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Accuracy vs. perturbation ({family}, {model_id})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_robustness_gap_heatmap(df: pd.DataFrame, acc_tbl: pd.DataFrame, model_id: str, out):
    plt = _ensure_mpl()
    sub = acc_tbl[(acc_tbl["model_id"] == model_id)]
    piv_ns = sub[sub["branch"] == "neurosymbolic"].pivot_table(
        index="calc_label" if "calc_label" in sub else "family", columns="level", values="accuracy")
    # gap by calculator: neurosymbolic - plain per calc x level
    d = df[df["model_id"] == model_id].copy()
    d["hit"] = d["correct"].astype(float)
    piv = d.pivot_table(index="calc_name", columns=["level"], values="hit", aggfunc="mean",
                        observed=True)
    ns = d[d["branch"] == "neurosymbolic"].pivot_table(index="calc_name", columns="level",
                                                        values="hit", aggfunc="mean")
    pl = d[d["branch"] == "plain_cot"].pivot_table(index="calc_name", columns="level",
                                                    values="hit", aggfunc="mean")
    gap = (ns - pl).reindex(columns=[l for l in LEVEL_SEQ if l in ns.columns])
    if gap.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(gap) + 1)))
    im = ax.imshow(gap.to_numpy(), cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(gap.columns)))
    ax.set_xticklabels(gap.columns)
    ax.set_yticks(range(len(gap.index)))
    ax.set_yticklabels([c[:28] for c in gap.index], fontsize=7)
    ax.set_title(f"Robustness gap (neuro-symbolic − plain), {model_id}")
    fig.colorbar(im, ax=ax, label="Δ accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_compute_matched(cm: pd.DataFrame, out):
    plt = _ensure_mpl()
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    markers = {"frontier_a": "o", "open_b": "s"}
    mlabel = {"frontier_a": "frontier", "open_b": "open"}
    colors = {"plain_cot": "#1f77b4", "scaled": "#2ca02c", "neurosymbolic": "#d62728"}
    # focus on the equation family (the headline comparison) to avoid clutter
    sub = cm[cm["family"] == "equation"] if "equation" in set(cm["family"]) else cm
    for _, r in sub.iterrows():
        ax.scatter(r["mean_tokens"], r["accuracy"],
                   marker=markers.get(r["model_id"], "o"), s=110,
                   color=colors.get(r["branch"], "gray"),
                   edgecolor="black", linewidth=0.6, zorder=3)
    ax.set_xlabel("mean tokens per item (compute)")
    ax.set_ylabel("accuracy")
    ax.set_title("Compute-matched: accuracy vs. tokens (equation family)")
    ax.grid(alpha=0.3)
    branch_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                             markeredgecolor="black", markersize=9, label=BRANCH_LABEL[b])
                      for b, c in colors.items()]
    model_handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
                            markeredgecolor="black", markersize=9, label=mlabel[k])
                     for k, m in markers.items()]
    leg1 = ax.legend(handles=branch_handles, loc="lower right", fontsize=8, title="branch")
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, loc="center right", fontsize=8, title="model")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_fail_modes(fm_tbl: pd.DataFrame, model_id: str, out):
    plt = _ensure_mpl()
    sub = fm_tbl[fm_tbl["model_id"] == model_id].sort_values("level_ord")
    if sub.empty:
        return None
    levels = sub["level"].tolist()
    modes = ["none", "formalization", "execution", "arithmetic"]
    colors = {"none": "#2ca02c", "formalization": "#d62728",
              "execution": "#ff7f0e", "arithmetic": "#9467bd"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bottom = np.zeros(len(sub))
    totals = sub[modes].sum(axis=1).to_numpy().clip(min=1)
    for m in modes:
        frac = sub[m].to_numpy() / totals
        ax.bar(levels, frac, bottom=bottom, label=m, color=colors[m])
        bottom += frac
    ax.set_ylabel("fraction of neuro-symbolic outputs")
    ax.set_xlabel("perturbation level")
    ax.set_title(f"Neuro-symbolic outcome attribution ({model_id})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
