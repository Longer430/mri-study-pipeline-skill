#!/usr/bin/env python3
"""Fail-soft staged inference for a time-limited hidden test (SKILL Section 8).

Each ensemble member runs as its own subprocess with a timeout and writes a CSV
(id column + finding columns). The runner:
  * skips a stage when the remaining budget is below its estimated time,
  * validates every output (ids match the expected list, no duplicates, finite values in [0, 1]),
  * blends the members that succeeded by per-finding percentile rank with weights
    renormalised over survivors,
  * optionally sets a canary column to a constant when any member failed, so a broken
    hidden run shows up as a visibly lower score instead of "the same score",
  * writes the submission atomically (temp file, validate, rename),
  * logs what ran, how long it took and why anything was dropped.

Plan file (JSON):
{
  "ids": "test_ids.csv",               # CSV with the id column, defines expected rows
  "id_col": "StudyInstanceUID",
  "findings": ["ACL", "MCL", ...],
  "budget_s": 30000,                    # total wall-clock budget (e.g. 8.3 h of 9 h)
  "canary": {"column": "ACL", "value": 0.5},   # optional
  "stages": [
    {"name": "coatnet", "cmd": ["python", "infer_a.py", "--out", "preds/a.csv"],
     "out": "preds/a.csv", "weight": 0.5, "timeout_s": 9000, "est_s": 7000},
    ...
  ]
}
  python scripts/staged_inference.py plan.json submission.csv
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def validate(df: pd.DataFrame, ids: pd.Index, findings: list[str], id_col: str) -> str | None:
    if id_col not in df.columns:
        return f"missing id column {id_col}"
    if df[id_col].duplicated().any():
        return "duplicate ids"
    missing_cols = [f for f in findings if f not in df.columns]
    if missing_cols:
        return f"missing columns {missing_cols}"
    got = set(df[id_col].astype(str))
    exp = set(ids)
    if got != exp:
        return f"id mismatch: {len(exp - got)} expected ids missing, {len(got - exp)} unexpected"
    vals = df[findings].to_numpy(float)
    if not np.isfinite(vals).all():
        return f"{int((~np.isfinite(vals)).sum())} non-finite values"
    if (vals < 0).any() or (vals > 1).any():
        return "values outside [0, 1]"
    return None


def pct_rank(x: pd.Series) -> pd.Series:
    return x.rank(method="average") / len(x)


def blend(results: dict[str, pd.DataFrame], weights: dict[str, float], findings: list[str]) -> pd.DataFrame:
    wsum = sum(weights[n] for n in results)
    out = None
    for name, df in results.items():
        part = df[findings].apply(pct_rank) * (weights[name] / wsum)
        out = part if out is None else out + part
    return out


def run(plan_path: str, out_path: str) -> int:
    plan = json.loads(Path(plan_path).read_text())
    id_col = plan.get("id_col", "StudyInstanceUID")
    findings = plan["findings"]
    ids = pd.Index(pd.read_csv(plan["ids"])[id_col].astype(str))
    t0, budget = time.time(), float(plan["budget_s"])
    results, log = {}, []
    for st in plan["stages"]:
        remaining = budget - (time.time() - t0)
        rec = {"stage": st["name"], "status": None, "seconds": 0.0, "reason": ""}
        if remaining < st.get("est_s", 0):
            rec.update(status="skipped", reason=f"remaining {remaining:.0f}s < estimate {st.get('est_s')}s")
            log.append(rec); continue
        start = time.time()
        try:
            proc = subprocess.run(st["cmd"], timeout=min(st.get("timeout_s", remaining), remaining),
                                  capture_output=True, text=True)
            rec["seconds"] = round(time.time() - start, 1)
            if proc.returncode != 0:
                rec.update(status="failed", reason=f"exit {proc.returncode}: {proc.stderr.strip()[-300:]}")
                log.append(rec); continue
        except subprocess.TimeoutExpired:
            rec.update(status="timeout", seconds=round(time.time() - start, 1), reason="timed out")
            log.append(rec); continue
        try:
            df = pd.read_csv(st["out"])
            df[id_col] = df[id_col].astype(str)
        except Exception as e:  # noqa: BLE001
            rec.update(status="failed", reason=f"unreadable output: {e}")
            log.append(rec); continue
        err = validate(df, ids, findings, id_col)
        if err:
            rec.update(status="invalid", reason=err)
            log.append(rec); continue
        results[st["name"]] = df.set_index(id_col).loc[ids]
        rec["status"] = "ok"
        log.append(rec)

    weights = {s["name"]: float(s.get("weight", 1.0)) for s in plan["stages"]}
    if not results:
        print(json.dumps(log, indent=2))
        print("no member succeeded; refusing to write a constant submission", file=sys.stderr)
        return 2
    sub = blend(results, weights, findings)
    canary = plan.get("canary")
    if canary and len(results) < len(plan["stages"]):
        sub[canary["column"]] = canary["value"]
    sub.index.name = id_col
    sub = sub.reset_index()
    err = validate(sub, ids, findings, id_col)
    if err:
        print(f"blend failed validation: {err}", file=sys.stderr)
        return 3
    tmp = Path(out_path).with_suffix(".tmp.csv")
    sub.to_csv(tmp, index=False)
    if validate(pd.read_csv(tmp, dtype={id_col: str}), ids, findings, id_col) is None:
        os.replace(tmp, out_path)
    used = {n: round(weights[n] / sum(weights[m] for m in results), 3) for n in results}
    print(json.dumps({"log": log, "weights_used": used, "canary_set": bool(canary and len(results) < len(plan["stages"]))},
                     indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(run(sys.argv[1], sys.argv[2]))
