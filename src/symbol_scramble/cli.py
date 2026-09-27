"""CLI entrypoints (SPEC §10)."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="sscramble")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="fetch MedCalc-Bench data")
    sub.add_parser("healthcheck", help="oracle reproduces all ground truth")
    sub.add_parser("audit", help="audit correctness gate: patch/quarantine/verify calculators")

    g = sub.add_parser("gen", help="generate perturbed variants")
    g.add_argument("--levels", default=None, help="comma-separated, e.g. L0,L1,L2")
    g.add_argument("--n", type=int, default=None, help="variants per (item, level)")
    g.add_argument("--calc", default=None, help="comma-separated calc_ids, or 'all'")
    g.add_argument("--max-seeds", type=int, default=3, help="seed items per calculator")
    g.add_argument("--config", default="default")
    g.add_argument("--append", action="store_true", help="merge into existing variants.parquet")

    r = sub.add_parser("run", help="run solver branches over variants")
    r.add_argument("--branches", default=None, help="default: config branches")
    r.add_argument("--models", default=None, help="comma-separated model ids")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--subset", type=int, default=None,
                   help="stratified subset of N variants (for cost-bounded frontier runs)")
    r.add_argument("--config", default="default")

    mq = sub.add_parser("medqa", help="run the MedQA boundary tier (M5)")
    mq.add_argument("--models", default=None)
    mq.add_argument("--n", type=int, default=60)
    mq.add_argument("--config", default="default")

    sub.add_parser("report", help="metrics + figures + report.md")

    args = p.parse_args(argv)

    if args.cmd == "download":
        from .data_loader import download_medcalc

        download_medcalc()
        print("download complete")
    elif args.cmd == "audit":
        from .audit import run_audit_gate

        r = run_audit_gate()
        print(f"patched: {r.patched}")
        print(f"quarantined: {list(r.quarantined)}")
        print(f"verified {len(r.verified)} formulas; failures: {r.failed}")
        sys.exit(1 if r.failed else 0)
    elif args.cmd == "healthcheck":
        from .healthcheck import run_healthcheck

        rep = run_healthcheck()
        print(f"healthcheck: {rep.reproduced}/{rep.total} reproduced ({rep.rate:.1%})")
        if rep.per_calc_fail:
            print("failures:", rep.per_calc_fail)
        sys.exit(0 if rep.rate == 1.0 else 1)
    elif args.cmd == "gen":
        from .runner import generate_variants

        levels = args.levels.split(",") if args.levels else None
        calc_ids = None if (not args.calc or args.calc == "all") else args.calc.split(",")
        vs = generate_variants(args.config, calc_ids, levels, args.n, args.max_seeds,
                               append=args.append)
        import collections

        by = collections.Counter(v.level for v in vs)
        print(f"generated {len(vs)} variants: {dict(by)}")
    elif args.cmd == "run":
        from .runner import run_experiment

        branches = args.branches.split(",") if args.branches else None
        models = args.models.split(",") if args.models else None
        run_experiment(args.config, branches, models, args.limit, args.subset)
    elif args.cmd == "medqa":
        from .medqa import run_medqa

        models = args.models.split(",") if args.models else None
        run_medqa(models=models, n=args.n, config_name=args.config)
    elif args.cmd == "report":
        from .report import build_report

        info = build_report()
        print(f"report built: {info['n_records']} records, {len(info['figures'])} figures")


if __name__ == "__main__":
    main()
