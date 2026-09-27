#!/usr/bin/env python3
"""Regenerate every number the paper reports, from the released result records.

No API calls, no network, no spend: this reads the run records in
data/results/rulearena/ and recomputes the macros, tables and figure the paper is
built from. If a number here disagrees with the paper, the paper is wrong.

    python reproduce.py            # regenerate and print the headline numbers
    python reproduce.py --check    # also fail if the test suite does not pass

Requires only the dependencies in pyproject.toml.
"""
import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
GENERATORS = [
    ("fill_ra_macros", "airline and tax accuracy, cost, ratios"),
    ("fill_ra_stats", "paired McNemar p-values and significance"),
    ("fill_ra_sweep", "the three airline difficulty tiers"),
    ("fill_ra_ci", "Tango score intervals"),
    ("fill_ra_ceiling", "the completion-ceiling correction"),
    ("fill_ra_extract", "unstructured input: extraction and end-to-end"),
    ("fill_ra_fresh", "generated inputs: the contamination control"),
    ("fill_ra_taxl1", "worked-example coverage on the harder tax tier"),
    ("fill_ra_scope", "how many cells favor each strategy"),
    ("cost_curve", "break-even and the cost curve"),
    ("loss_report", "which cases each strategy trades"),
]

# The two figures rebuild the same way, but they are the only step needing matplotlib,
# so a missing plotting stack skips them rather than failing the run: the numbers are
# the reproducible claim, the figures are only how the paper draws them.
FIGURES = [
    ("make_paper_forest", "Figure 3: paired differences with Tango intervals"),
    ("make_cost_curve_figure", "Figure 4: the cost advantage as a curve"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="also run the test suite")
    args = ap.parse_args()

    # the generators write here; a fresh clone has no paper/ directory
    (ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper" / "figures").mkdir(parents=True, exist_ok=True)

    env = {"PYTHONPATH": str(ROOT / "src")}
    failed = []
    for mod, what in GENERATORS:
        r = subprocess.run([sys.executable, str(ROOT / "tools" / f"{mod}.py")],
                           capture_output=True, text=True, cwd=ROOT,
                           env={**dict(__import__("os").environ), **env})
        ok = r.returncode == 0
        print(f"  {'ok  ' if ok else 'FAIL'}  {mod:18s} {what}")
        if not ok:
            failed.append((mod, r.stderr.strip().splitlines()[-1:] or ["?"]))

    try:
        __import__("matplotlib")
        for mod, what in FIGURES:
            r = subprocess.run([sys.executable, str(ROOT / "tools" / f"{mod}.py")],
                               capture_output=True, text=True, cwd=ROOT,
                               env={**dict(__import__("os").environ), **env})
            ok = r.returncode == 0
            print(f"  {'ok  ' if ok else 'FAIL'}  {mod:18s} {what}")
            if not ok:
                failed.append((mod, r.stderr.strip().splitlines()[-1:] or ["?"]))
    except ImportError:
        print("  skip  figures            (matplotlib not installed)")

    if failed:
        print("\nfailed generators:")
        for mod, err in failed:
            print(f"  {mod}: {err[0]}")
        return 1

    print("\nHeadline numbers, recomputed:")
    mac = {}
    for f in (ROOT / "paper").glob("*.tex"):
        for line in f.read_text().splitlines():
            m = __import__("re").match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line.strip())
            if m:
                mac[m.group(1)] = m.group(2)
    for label, key in [
        ("cost advantage", "raRatioAllLo"), ("      .. to", "raRatioAllHi"),
        ("break-even (cases)", "raCvBreakLo"), ("      .. to", "raCvBreakHi"),
        ("cells reported", "raScopeCells"),
        ("cells favoring formalize-once", "raScopeWins"),
        ("cells favoring per-case", "raScopeLosses"),
    ]:
        if key in mac:
            print(f"  {label:32s} {mac[key]}")

    if args.check:
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=ROOT)
        if r.returncode != 0:
            return r.returncode
    print("\nRegenerated from data/results/rulearena/ with no network access.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
