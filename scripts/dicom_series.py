#!/usr/bin/env python3
"""DICOM series helpers: slice ordering, header-based series typing, laterality.

Works on pydicom Datasets or on any object/dict exposing the same attribute names,
so the logic can be unit-tested without real data. Nothing here reads pixel data.

Rules follow the RSNA 2026 Knee write-ups (22nd, 49th, 76th): never trust
SeriesDescription when the header can decide; sort by physical position; only
the sagittal x-position carries the side. Audit every rule on your own data:
vendor quirks differ (e.g. a host flag meaning "fluid is bright" rather than
"fat suppressed"), and the thresholds below are starting points, not constants.

  python scripts/dicom_series.py /path/to/study_dir     # prints one row per series (needs pydicom)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


def _get(ds, name, default=None):
    if isinstance(ds, dict):
        return ds.get(name, default)
    return getattr(ds, name, default)


def _as_floats(v):
    if v is None:
        return None
    try:
        return [float(x) for x in v]
    except TypeError:
        return [float(v)]


def _text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)) or (hasattr(v, "__iter__") and not isinstance(v, (str, bytes))):
        return " ".join(str(x) for x in v).upper()
    return str(v).upper()


# --------------------------------------------------------------- geometry
def slice_normal(ds) -> np.ndarray | None:
    iop = _as_floats(_get(ds, "ImageOrientationPatient"))
    if not iop or len(iop) != 6:
        return None
    row, col = np.array(iop[:3]), np.array(iop[3:])
    n = np.cross(row, col)
    norm = np.linalg.norm(n)
    return n / norm if norm else None


def plane_of(normal: np.ndarray) -> str:
    """Patient (LPS) axes: x -> sagittal, y -> coronal, z -> axial."""
    return ["sagittal", "coronal", "axial"][int(np.argmax(np.abs(normal)))]


def sort_slices(slices: list) -> list:
    """Order slices along the plane normal (ImagePositionPatient . normal), never by
    InstanceNumber or file name. The direction is fixed to increase along the dominant
    patient axis, so different scanners give the same anatomical order. Drops slices whose orientation differs from the
    majority (localisers mixed into a series)."""
    normals = [slice_normal(s) for s in slices]
    valid = [(s, n) for s, n in zip(slices, normals) if n is not None and _get(s, "ImagePositionPatient")]
    if not valid:
        return list(slices)
    ref = valid[len(valid) // 2][1]
    # make the order scanner-independent: increase along the dominant patient axis
    # (x: right->left, y: anterior->posterior, z: inferior->superior in LPS)
    ref = ref * np.sign(ref[int(np.argmax(np.abs(ref)))])
    keep = [(s, n) for s, n in valid if abs(float(np.dot(n, ref))) > 0.99]
    keyed = [(float(np.dot(np.array(_as_floats(_get(s, "ImagePositionPatient"))), ref)), i, s)
             for i, (s, _) in enumerate(keep)]
    return [s for _, _, s in sorted(keyed)]


# ------------------------------------------------------------ series typing
@dataclass
class SeriesAttrs:
    plane: str | None
    oblique: bool
    fat_suppressed: bool
    weighting: str          # ZTE / STIR / GRE / T1 / PD_T2 / unknown
    is_3d: bool
    te: float | None
    tr: float | None
    n_slices: int
    source: str             # "header" or "name_fallback"


def series_attributes(slices: list, host_fs_flag: bool | None = None,
                      oblique_threshold: float = 0.9) -> SeriesAttrs:
    """Derive series attributes from header fields of the (sorted) slices."""
    ds = slices[len(slices) // 2]
    n = slice_normal(ds)
    plane = plane_of(n) if n is not None else None
    oblique = bool(n is not None and np.max(np.abs(n)) < oblique_threshold)
    te = _as_floats(_get(ds, "EchoTime"))
    tr = _as_floats(_get(ds, "RepetitionTime"))
    te = te[0] if te else None
    tr = tr[0] if tr else None
    seq = _text(_get(ds, "ScanningSequence")) + " " + _text(_get(ds, "SequenceVariant"))
    opts = _text(_get(ds, "ScanOptions"))
    vendor = _text(_get(ds, "Manufacturer"))
    acq = _text(_get(ds, "MRAcquisitionType"))
    thick = _as_floats(_get(ds, "SliceThickness"))
    is_3d = acq == "3D" or (acq == "" and thick is not None and thick[0] <= 1.2)
    ti = _as_floats(_get(ds, "InversionTime"))
    inversion = "IR" in seq.split() or (ti is not None and ti[0] > 0)
    gradient = "GR" in seq.split()

    fs = bool(host_fs_flag) or any(k in opts for k in ("FS", "WE", "FSA", "SFS")) or inversion
    # vendor quirk reported in the write-ups: water-selective 3D GRE without an FS hint
    if "PHILIPS" in vendor and is_3d and gradient and tr is not None and tr >= 10:
        fs = True

    source = "header"
    if te is not None and te < 1:
        weighting = "ZTE"
    elif inversion:
        weighting = "STIR"
    elif gradient:
        weighting = "GRE"
    elif te is not None and tr is not None:
        weighting = "T1" if te < 30 and tr < (800 if is_3d else 1200) else "PD_T2"
    else:
        name = _text(_get(ds, "SeriesDescription"))
        source = "name_fallback"
        weighting = ("T1" if "T1" in name else "PD_T2" if any(k in name for k in ("PD", "T2", "DP")) else "unknown")
        fs = fs or any(k in name for k in ("FS", "FAT", "SPAIR", "SPIR", "STIR", "TIRM", "DIXON_W", "_W"))
    return SeriesAttrs(plane, oblique, fs, weighting, is_3d, te, tr, len(slices), source)


# ---------------------------------------------------------------- laterality
def laterality(series: dict[str, list], min_offset_mm: float = 20.0) -> tuple[str | None, str]:
    """Return ('L'|'R'|None, how). Order: Laterality / ImageLaterality tag, then the
    sign of the mean sagittal x position (LPS: +x is the patient's left). Only the
    sagittal series carry the side; coronal/axial offsets do not. If |x| is small
    the knee sat at isocentre and the sign says nothing -> None (use an image
    classifier). Validate the sign rule on studies that have the tag first."""
    for slices in series.values():
        for s in slices[:1]:
            for tag in ("Laterality", "ImageLaterality"):
                v = _text(_get(s, tag)).strip()
                if v in ("L", "R"):
                    return v, f"tag:{tag}"
    xs = []
    for slices in series.values():
        n = slice_normal(slices[0]) if slices else None
        if n is not None and plane_of(n) == "sagittal":
            xs += [_as_floats(_get(s, "ImagePositionPatient"))[0] for s in slices
                   if _get(s, "ImagePositionPatient") is not None]
    if xs and abs(np.mean(xs)) >= min_offset_mm:
        return ("L" if np.mean(xs) > 0 else "R"), "sagittal_x_sign"
    return None, "undetermined"


# ---------------------------------------------------------------------- CLI
def _load_study(path: Path) -> dict[str, list]:
    import pydicom  # only needed for the CLI
    series = defaultdict(list)
    for f in sorted(path.rglob("*")):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue  # a corrupt file costs a file, not the study
        series[str(_get(ds, "SeriesInstanceUID", f.parent.name))].append(ds)
    return series


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    study = _load_study(Path(sys.argv[1]))
    side, how = laterality(study)
    print(f"laterality: {side} ({how})")
    for uid, sl in study.items():
        a = series_attributes(sort_slices(sl))
        print(uid[-12:], _text(_get(sl[0], "SeriesDescription"))[:30].ljust(30), asdict(a))


if __name__ == "__main__":
    main()
