"""Write each cell's no-answer flags into its result record.

A plain call that exhausts its completion ceiling returns no text. Those cases are
excluded from the paired test rather than scored wrong, so which cases they are is part
of the result -- but until now it was recoverable only by replaying the content-addressed
call cache and re-deriving the flags. That made a 26 MB cache a hidden prerequisite for
getting the paper's numbers right: without it every cell silently counted its truncated
cases as wrong answers, which flips at least one cell's significance.

This backfills the flags into the records once, so the records stand on their own and
the cache becomes an optimization for re-running rather than a requirement for reading.

Run:  .venv/bin/python tools/backfill_noanswer.py [--write]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
RES = ROOT / "data" / "results" / "rulearena"


def main(write=False):
    from noanswer_report import no_answer_flags
    from ra_cells import RAISED, files_for

    touched = 0
    for domain, level in (("airline", 0), ("airline", 1), ("airline", 2), ("tax", 0)):
        for fname in files_for(domain, level):
            fp = RES / fname
            if not fp.exists():
                continue
            out, changed = [], False
            for line in fp.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("n") == 90 and "per_case_no_answer" not in r:
                    flags = no_answer_flags(r["model"], r.get("level", 0),
                                            RAISED.get((r["model"], domain, level)),
                                            domain=domain)
                    n = len(r.get("per_case_plain", []))
                    r["per_case_no_answer"] = [int(bool(flags[i])) if i < len(flags)
                                               else 0 for i in range(n)]
                    changed = True
                    if sum(r["per_case_no_answer"]):
                        print(f"  {fname:28s} {r['model']:16s} "
                              f"{sum(r['per_case_no_answer'])} no-answer case(s)")
                out.append(json.dumps(r))
            if changed:
                touched += 1
                if write:
                    fp.write_text("\n".join(out) + "\n")
    print(f"\n{'wrote' if write else 'would write'} {touched} file(s)")
    if not write:
        print("re-run with --write to apply")


if __name__ == "__main__":
    main("--write" in sys.argv)
