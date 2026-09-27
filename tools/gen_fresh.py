"""Fresh-case generator for a contamination-robustness check.

For each RuleArena domain we produce NOVEL structured inputs by keeping a real case's
structure but resampling its field values within the ranges/sets observed across the
public comp_0 set, then relabel with the deterministic oracle. These inputs do not
appear in any public dataset, so accuracy on them cannot be explained by memorization
of RuleArena's released (input->answer) pairs. This is the GSM-Symbolic / Functional-
Benchmarks recipe (regenerate values, recompute the answer); we do not claim it as
novel -- it is a robustness control for our own result.

Nothing here mutates existing files. Deterministic given `seed`.
Returns lightweight objects exposing `.entities` and `.gt_answer` (all core_experiment
needs), so no dependency on the Variant schema.
"""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

from symbol_scramble import paths
from symbol_scramble.rulearena import oracle

AIR_PATH = paths.RAW / "rulearena" / "airline" / "comp_0.jsonl"
TAX_PATH = paths.RAW / "rulearena" / "tax" / "comp_0.json"


def _airline_seeds():
    return [json.loads(l)["info"] for l in open(AIR_PATH) if l.strip()]


def _tax_seeds():
    recs = json.load(open(TAX_PATH))
    return [{k: v for k, v in r["pydantic"].items()
             if k not in oracle.TAX_IGNORED_FIELDS} for r in recs]


def _boundary_rate(seeds, size_edges, wt_edges, tol=1):
    """Fraction of public checked bags sitting within tol of a rule boundary."""
    tot = hit = 0
    for s_ in seeds:
        for b in s_["bag_list"][1:]:
            tot += 1
            if (any(abs(sum(b["size"]) - e) <= tol for e in size_edges)
                    or any(abs(b["weight"] - e) <= tol for e in wt_edges)):
                hit += 1
    return hit / max(tot, 1)


def _ranges(vals):
    """min/max for a list of numbers."""
    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return (min(nums), max(nums)) if nums else (0, 0)


def _fresh_airline(seeds, n, rng):
    classes = sorted({s["customer_class"] for s in seeds})
    routines = sorted({s["routine"] for s in seeds})
    prices = _ranges([s["base_price"] for s in seeds])
    # bag dimension / weight ranges across all checked bags
    dims = _ranges([d for s in seeds for b in s["bag_list"][1:] for d in b["size"]])
    wts = _ranges([b["weight"] for s in seeds for b in s["bag_list"][1:]])
    names = sorted({b["name"] for s in seeds for b in s["bag_list"]})
    # The oversize rules key on the SUM of a bag's dimensions (linear inches), and
    # the published fee schedule prices only the sums the public set exercises.
    # Constrain the resampled sum to the public per-bag envelope: three i.i.d.
    # per-dimension draws would otherwise leave it in ~77% of cases and land 56%
    # of cases in an unpriced bracket (>115 linear inches).
    sums = _ranges([sum(b["size"]) for s in seeds for b in s["bag_list"][1:]])

    # RuleArena's public cases are adversarial by construction: 48% of checked bags
    # sit within a unit of a rule boundary (62/65 linear inches, 50/70/100 lb),
    # which is exactly where a per-case reasoner drops a condition or picks the
    # wrong bracket. Sampling uniformly inside the envelope reproduces the
    # aggregate profile -- bags per case, oversize rate, overweight rate -- but
    # lands on a boundary only 12% of the time, which makes the cases markedly
    # easier and leaves both strategies near 0.99 with no headroom to differ.
    # Fresh inputs have to match that boundary density or they test difficulty
    # rather than contamination.
    SIZE_EDGES, WT_EDGES = (62, 65), (50, 70, 100)
    edge_rate = _boundary_rate(seeds, SIZE_EDGES, WT_EDGES)

    def _draw_size(force_edge=False):
        if force_edge:                      # place the SUM on a size boundary
            target = rng.choice(SIZE_EDGES) + rng.choice([-1, 0, 0, 1])
            if sums[0] <= target <= sums[1]:
                a = rng.randint(max(dims[0], target - 2 * dims[1]), min(dims[1], target - 2 * dims[0]))
                rest = target - a
                b_ = rng.randint(max(dims[0], rest - dims[1]), min(dims[1], rest - dims[0]))
                return [a, b_, rest - b_]
        while True:
            size = [rng.randint(*dims), rng.randint(*dims), rng.randint(*dims)]
            if sums[0] <= sum(size) <= sums[1]:
                return size

    def _draw_bag(on_edge):
        """One bag. If it should sit on a boundary, put it on exactly one of them."""
        if not on_edge:
            return {"size": _draw_size(), "weight": _draw_weight()}
        if rng.random() < 0.5:
            sz = _draw_size(force_edge=True)
            if any(abs(sum(sz) - e) <= 1 for e in SIZE_EDGES):
                return {"size": sz, "weight": _draw_weight()}
            return {"size": sz, "weight": _draw_weight(force_edge=True)}
        return {"size": _draw_size(), "weight": _draw_weight(force_edge=True)}

    def _draw_weight(force_edge=False):
        if force_edge:
            w = rng.choice(WT_EDGES) + rng.choice([-1, 0, 0, 1])
            if wts[0] <= w <= wts[1]:
                return w
        return rng.randint(*wts)
    out = []
    while len(out) < n:
        tmpl = rng.choice(seeds)
        info = {
            "base_price": rng.randint(*prices),
            "customer_class": rng.choice(classes),
            "routine": rng.choice(routines),
            "direction": rng.choice([0, 1]),
            "bag_list": [],
        }
        # keep the template's bag count/structure; resample every checked bag's numbers
        for i, b in enumerate(tmpl["bag_list"]):
            if i == 0:  # carry-on / personal item: keep small, free
                info["bag_list"].append(dict(b))
                continue
            info["bag_list"].append({
                "id": b.get("id", i + 1),
                "name": rng.choice(names),
                # match the public boundary density rather than sampling flat
                **_draw_bag(rng.random() < edge_rate),
            })
        try:
            gt = oracle.airline_answer(info)
        except Exception:  # noqa: BLE001 - skip anything the oracle cannot label
            continue
        out.append(SimpleNamespace(entities=info, gt_answer=gt))
    return out


def _fresh_tax(seeds, n, rng):
    statuses = sorted({s["filing_status"] for s in seeds})
    # dollar fields = every numeric (non-bool) key; ranges observed across comp_0
    money_keys = sorted({k for s in seeds for k, v in s.items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)
                         and k not in ("age", "spouse_age", "num_qualifying_children",
                                       "num_other_dependents")})
    ranges = {k: _ranges([s.get(k, 0) for s in seeds]) for k in money_keys}
    out = []
    while len(out) < n:
        tmpl = rng.choice(seeds)
        p = dict(tmpl)  # keep booleans/name/structure (comp_0 basic: self_employed/itemized False)
        p["filing_status"] = rng.choice(statuses)
        p["age"] = rng.randint(18, 90)
        if "spouse_age" in p:
            p["spouse_age"] = rng.randint(18, 90)
        if "blind" in p:
            p["blind"] = rng.random() < 0.15
        if "num_qualifying_children" in p:
            p["num_qualifying_children"] = rng.randint(0, 3)
        if "num_other_dependents" in p:
            p["num_other_dependents"] = rng.randint(0, 2)
        for k, (lo, hi) in ranges.items():
            if k in p:
                p[k] = float(rng.randint(int(lo), int(max(lo, hi))))
        try:
            gt = oracle.tax_answer(p)
        except Exception:  # noqa: BLE001
            continue
        out.append(SimpleNamespace(entities=p, gt_answer=gt))
    return out


def _matched_airline(seeds, n, rng):
    """Novel airline cases drawn from the PUBLIC set's own marginals.

    The first generator designed its own value distribution and matched only the
    fraction of bags sitting exactly on a rule boundary. That is not enough. The
    public bags cluster tightly just past the size threshold (median 65 linear inches
    against a 62-inch edge) while the generated ones sprawled to a median of 86, where
    oversize is obvious rather than borderline; public trips are 39% U.S.-domestic
    against 6% generated, and domestic is the harder route for per-case reasoning
    (0.83 vs 0.95 for Opus); public fares run about half the generated ones. Those
    three gaps made the generated cases easier for a per-case reasoner, which is
    exactly the quantity a contamination control is trying to hold fixed -- the first
    run's near-zero gaps could not be read as evidence about memorization.

    So every field here is resampled from the public empirical distribution and then
    jittered, rather than drawn from a designed one. Novelty comes from recombining
    and perturbing values across cases, not from widening their range. The answer is
    recomputed by the oracle, so a memorized (input, answer) pair does not transfer.
    """
    pool_bags = [b for s in seeds for b in s["bag_list"][1:]]
    pool_carry = [s["bag_list"][0] for s in seeds]
    pool_price = [s["base_price"] for s in seeds]
    pool_cls = [s["customer_class"] for s in seeds]
    pool_route = [(s["routine"], s["direction"]) for s in seeds]
    pool_nbag = [len(s["bag_list"]) for s in seeds]
    public = {json.dumps(s, sort_keys=True) for s in seeds}

    def jitter_bag(b):
        """Perturb a real bag. The jitter is small relative to the spread but large
        relative to the thresholds, so a bag can cross an edge -- which changes the
        answer and is the point -- while the distribution's shape is preserved."""
        sz = [max(1, x + rng.randint(-2, 2)) for x in b["size"]]
        return {"name": b["name"], "size": sz,
                "weight": max(1, b["weight"] + rng.randint(-3, 3))}

    out, guard = [], 0
    while len(out) < n and guard < n * 200:
        guard += 1
        routine, direction = rng.choice(pool_route)
        bags = [dict(rng.choice(pool_carry))]
        for _ in range(rng.choice(pool_nbag) - 1):
            bags.append(jitter_bag(rng.choice(pool_bags)))
        for i, b in enumerate(bags, 1):
            b["id"] = i
        price = rng.choice(pool_price)
        info = {"base_price": max(20, int(price * rng.uniform(0.85, 1.15))),
                "customer_class": rng.choice(pool_cls), "routine": routine,
                "direction": direction, "bag_list": bags}
        if json.dumps(info, sort_keys=True) in public:
            continue                      # never hand back a public case
        try:
            gt = oracle.airline_answer(info)
        except Exception:                 # noqa: BLE001 - reject anything unpriceable
            continue
        out.append(SimpleNamespace(entities=info, gt_answer=gt))
    return out


def fresh_cases(domain: str, n: int, seed: int = 20260715, matched: bool = True):
    """matched=True resamples from the public set's own marginals (see
    _matched_airline). matched=False is the superseded designed-distribution
    generator, kept so the withdrawn run stays reproducible."""
    rng = random.Random(seed)
    if domain == "airline":
        return (_matched_airline if matched else _fresh_airline)(
            _airline_seeds(), n, rng)
    if domain == "tax":
        return _fresh_tax(_tax_seeds(), n, rng)
    raise ValueError(domain)


if __name__ == "__main__":
    for dom in ("airline", "tax"):
        cs = fresh_cases(dom, 8)
        print(f"{dom}: {len(cs)} fresh cases; sample answers:",
              [round(float(c.gt_answer), 2) for c in cs[:5]])
