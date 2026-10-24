#!/usr/bin/env python3
"""Self-test for dicom_series.py and staged_inference.py with synthetic inputs
(mock DICOM headers as dicts, fake ensemble members as tiny scripts). mil_head.py
needs torch; run `python scripts/mil_head.py` separately for its shape test.

  python scripts/selftest_pipeline.py --work selftest_pipeline
"""
import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dicom_series as dsr  # noqa: E402

FAILED = 0


def check(name, cond, detail=""):
    global FAILED
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    FAILED += 0 if cond else 1


def mock_series(plane, n=12, x0=60.0, spacing=3.0, shuffle=True, **hdr):
    iop = {"sagittal": [0, 1, 0, 0, 0, -1], "coronal": [1, 0, 0, 0, 0, -1], "axial": [1, 0, 0, 0, 1, 0]}[plane]
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        pos = [x0, -20.0, 30.0]
        pos[axis] += i * spacing
        out.append(dict(ImageOrientationPatient=iop, ImagePositionPatient=pos, InstanceNumber=n - i, idx=i, **hdr))
    if shuffle:
        rng.shuffle(out)
    return out


def test_dicom():
    s = mock_series("sagittal", EchoTime=35, RepetitionTime=3000, ScanningSequence="SE", ScanOptions="FS")
    ordered = dsr.sort_slices(s)
    check("sort: physical order, not InstanceNumber", [d["idx"] for d in ordered] == list(range(12)))
    loc = dict(ImageOrientationPatient=[1, 0, 0, 0, 1, 0], ImagePositionPatient=[0, 0, 0], idx=99)
    check("sort: drops a localiser with another orientation", all(d["idx"] != 99 for d in dsr.sort_slices(s + [loc])))

    a = dsr.series_attributes(ordered)
    check("typing: sagittal PD/T2 fat-suppressed from header", (a.plane, a.weighting, a.fat_suppressed) == ("sagittal", "PD_T2", True))
    t1 = dsr.series_attributes(mock_series("coronal", EchoTime=12, RepetitionTime=550, ScanningSequence="SE",
                                           SeriesDescription="PD_FS_wrong_name"))
    check("typing: header beats a misleading name (T1, not FS)", (t1.weighting, t1.fat_suppressed) == ("T1", False))
    stir = dsr.series_attributes(mock_series("coronal", EchoTime=40, RepetitionTime=4000, ScanningSequence="SE IR",
                                             InversionTime=150))
    check("typing: inversion recovery -> STIR and fat-suppressed", (stir.weighting, stir.fat_suppressed) == ("STIR", True))
    gre = dsr.series_attributes(mock_series("sagittal", EchoTime=5, RepetitionTime=20, ScanningSequence="GR",
                                            Manufacturer="Philips", MRAcquisitionType="3D"))
    check("typing: Philips 3D GRE with TR >= 10 counted as fat-suppressed", gre.fat_suppressed and gre.weighting == "GRE")
    zte = dsr.series_attributes(mock_series("sagittal", EchoTime=0.02, RepetitionTime=600))
    check("typing: TE < 1 ms -> ZTE", zte.weighting == "ZTE")
    fb = dsr.series_attributes(mock_series("axial", SeriesDescription="AX PD FS"))
    check("typing: name used only as fallback", fb.source == "name_fallback" and fb.fat_suppressed)
    obl = mock_series("coronal", shuffle=False)
    for d in obl:
        d["ImageOrientationPatient"] = [0.8, 0.6, 0, 0, 0, -1]
    check("typing: oblique flagged", dsr.series_attributes(obl).oblique)

    tagged = {"s1": mock_series("sagittal", x0=-80, Laterality="L")}
    check("laterality: tag wins", dsr.laterality(tagged) == ("L", "tag:Laterality"))
    left = {"s1": mock_series("sagittal", x0=70), "c1": mock_series("coronal", x0=-200)}
    check("laterality: sagittal x sign (+x = patient left), coronal ignored", dsr.laterality(left)[0] == "L")
    centre = {"s1": mock_series("sagittal", x0=-15)}
    check("laterality: near isocentre -> undetermined", dsr.laterality(centre)[0] is None)


def test_staged(work: Path):
    work.mkdir(parents=True, exist_ok=True)
    findings = ["ACL", "Effusion"]
    ids = [f"T{i:03d}" for i in range(50)]
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(work / "ids.csv", index=False)
    rng = np.random.default_rng(1)
    good = pd.DataFrame({"StudyInstanceUID": ids, "ACL": rng.random(50), "Effusion": rng.random(50)})
    good.to_csv(work / "good_src.csv", index=False)
    bad = good.copy(); bad.loc[3, "ACL"] = np.nan
    bad.to_csv(work / "nan_src.csv", index=False)
    scripts = {
        "copy.py": "import shutil,sys; shutil.copy(sys.argv[1], sys.argv[2])",
        "slow.py": "import time; time.sleep(30)",
        "crash.py": "raise SystemExit('boom')",
    }
    for n, code in scripts.items():
        (work / n).write_text(textwrap.dedent(code))
    py = sys.executable
    stages = [
        {"name": "good", "cmd": [py, str(work / "copy.py"), str(work / "good_src.csv"), str(work / "a.csv")],
         "out": str(work / "a.csv"), "weight": 0.5, "timeout_s": 20, "est_s": 0},
        {"name": "nan", "cmd": [py, str(work / "copy.py"), str(work / "nan_src.csv"), str(work / "b.csv")],
         "out": str(work / "b.csv"), "weight": 0.2, "timeout_s": 20, "est_s": 0},
        {"name": "slow", "cmd": [py, str(work / "slow.py")], "out": str(work / "c.csv"), "weight": 0.2,
         "timeout_s": 2, "est_s": 0},
        {"name": "crash", "cmd": [py, str(work / "crash.py")], "out": str(work / "d.csv"), "weight": 0.1,
         "timeout_s": 20, "est_s": 0},
        {"name": "too_big", "cmd": [py, "-c", "print(1)"], "out": str(work / "e.csv"), "weight": 0.1,
         "timeout_s": 20, "est_s": 10_000},
    ]
    plan = {"ids": str(work / "ids.csv"), "findings": findings, "budget_s": 60,
            "canary": {"column": "ACL", "value": 0.5}, "stages": stages}
    (work / "plan.json").write_text(json.dumps(plan))
    sub = work / "submission.csv"
    r = subprocess.run([py, str(HERE / "staged_inference.py"), str(work / "plan.json"), str(sub)],
                       capture_output=True, text=True)
    report = json.loads(r.stdout)
    status = {x["stage"]: x["status"] for x in report["log"]}
    check("staged: exit 0 with one surviving member", r.returncode == 0, r.stderr.strip()[-200:])
    check("staged: invalid / timeout / crash / over-budget all caught",
          status == {"good": "ok", "nan": "invalid", "slow": "timeout", "crash": "failed", "too_big": "skipped"},
          str(status))
    out = pd.read_csv(sub)
    check("staged: weights renormalised over survivors", report["weights_used"] == {"good": 1.0})
    check("staged: canary column set because members failed", out["ACL"].nunique() == 1 and out["ACL"].iloc[0] == 0.5)
    check("staged: submission complete and in [0, 1]", len(out) == 50 and out["Effusion"].between(0, 1).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="selftest_pipeline")
    work = Path(ap.parse_args().work)
    test_dicom()
    test_staged(work)
    print("FAILED" if FAILED else "ALL CHECKS PASSED", f"({FAILED} failures)")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
