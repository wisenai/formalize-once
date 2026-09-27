"""Download + load MedCalc-Bench seed items (IMPLEMENTATION_SPEC §5)."""
from __future__ import annotations

import ast
import json
import urllib.request
from typing import Optional

import pandas as pd
import yaml

from . import paths
from .schemas import SeedItem

_GH = "https://raw.githubusercontent.com/ncbi-nlp/MedCalc-Bench/main"
_API_TREE = "https://api.github.com/repos/ncbi-nlp/MedCalc-Bench/git/trees/main?recursive=1"


def _fetch(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        dest.write_bytes(r.read())


def download_medcalc(force: bool = False) -> None:
    """Fetch the test CSV + all calculator_implementations files. Idempotent."""
    paths.ensure_dirs()
    paths.CALC_IMPL.mkdir(parents=True, exist_ok=True)
    if force or not paths.MEDCALC_CSV.exists():
        _fetch(f"{_GH}/datasets/test_data.csv", paths.MEDCALC_CSV)
    with urllib.request.urlopen(_API_TREE, timeout=60) as r:
        tree = json.load(r)["tree"]
    files = [
        t["path"].split("/", 1)[1]
        for t in tree
        if t["type"] == "blob" and t["path"].startswith("calculator_implementations/")
    ]
    for fname in files:
        dest = paths.CALC_IMPL / fname
        if force or not dest.exists():
            _fetch(f"{_GH}/calculator_implementations/{fname}", dest)


def load_family_map() -> dict[int, dict]:
    with open(paths.CONFIGS / "calculator_family.yaml") as f:
        return yaml.safe_load(f)["calculators"]


def load_seed_items(
    families: tuple[str, ...] = ("equation", "score"),
    calc_ids: Optional[list[str]] = None,
    max_per_calc: Optional[int] = None,
) -> list[SeedItem]:
    """Parse the CSV into validated SeedItems, keeping only in-scope families."""
    fam_map = load_family_map()
    df = pd.read_csv(paths.MEDCALC_CSV)
    items: list[SeedItem] = []
    per_calc: dict[str, int] = {}
    quarantined = 0
    for _, r in df.iterrows():
        cid = str(int(r["Calculator ID"]))
        fam_info = fam_map.get(int(cid))
        if fam_info is None:
            continue
        family = fam_info["family"]
        if family not in families:
            quarantined += 1
            continue
        if calc_ids is not None and cid not in calc_ids:
            continue
        if max_per_calc is not None and per_calc.get(cid, 0) >= max_per_calc:
            continue
        try:
            entities = ast.literal_eval(r["Relevant Entities"])
        except (ValueError, SyntaxError):
            continue
        lower = _num(r.get("Lower Limit"))
        upper = _num(r.get("Upper Limit"))
        gt = _coerce_gt(r["Ground Truth Answer"], r["Output Type"])
        try:
            item = SeedItem(
                row_id=int(r["Row Number"]),
                calc_id=cid,
                calc_name=str(r["Calculator Name"]),
                category=str(r["Category"]),
                output_type=str(r["Output Type"]),
                family=family,
                note=str(r["Patient Note"]),
                question=str(r["Question"]),
                entities=entities,
                gt_answer=gt,
                lower=lower,
                upper=upper,
            )
        except Exception:  # noqa: BLE001 - quarantine malformed rows
            continue
        items.append(item)
        per_calc[cid] = per_calc.get(cid, 0) + 1
    return items


def _num(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except (ValueError, TypeError):
        return None


def _coerce_gt(x, output_type: str):
    if output_type == "integer":
        try:
            return int(round(float(x)))
        except (ValueError, TypeError):
            return str(x)
    if output_type == "decimal":
        try:
            return float(x)
        except (ValueError, TypeError):
            return str(x)
    return str(x)
