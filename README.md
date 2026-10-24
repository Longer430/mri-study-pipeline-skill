# MRI Study Pipeline

An agent skill (Claude / Codex `SKILL.md` format) for the **image side of study-level MRI classifiers**: multi-series studies, several findings per study, no slice-level labels, and offline, time-limited inference on a hidden test set.

It walks an agent through:

- **DICOM to arrays:** physical slice ordering, defensive decoding, caching next to the data
- **Laterality and orientation:** tag → sagittal position → classifier; no flip TTA after canonicalisation
- **Series typing:** plane, obliquity, fat suppression and weighting from header fields, never trusting series names
- **Input geometry:** physical crops (130–160 mm at ~0.3–0.4 mm/px), 2.5D windows from physical neighbours, reading all slices
- **Study model:** shared 2D backbone, zero-initialised window mixer, per-finding attention-MIL pooling
- **Anatomy:** ROI boxes, landmark crops, compartment classifiers, zoom on weak findings
- **Ensembling and inference engineering:** diversity that helps, stage timeouts, canaries, graceful degradation
- **What did not help:** a list of ideas several teams tried without gain

Companion skill for the label side: [medical-report-label-refinement](https://github.com/Longer430/medical-report-label-refinement-skill).

## Origin

Distilled from the post-competition write-ups of the [RSNA 2026 Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle competition. `references/rsna-knee-writeups.md` summarises each team's image-side choices in my own words, with links to the original posts. The code in `scripts/` is an independent implementation of the described methods, not copied from any team's repository. No competition data, images, labels or weights are included.

## Repository layout

```
SKILL.md                          the skill (what the agent reads)
references/rsna-knee-writeups.md  per-team digest with reported numbers and links
scripts/dicom_series.py           slice ordering, header-based series typing, laterality (+ CLI)
scripts/mil_head.py               zero-init window mixer + per-finding attention-MIL head (PyTorch)
scripts/staged_inference.py       fail-soft staged inference runner for a time-limited hidden test
scripts/selftest_pipeline.py      synthetic checks for dicom_series.py and staged_inference.py
agents/openai.yaml                Codex display metadata
```

## Quick check

```bash
pip install -r requirements.txt
python scripts/selftest_pipeline.py --work /tmp/selftest_pipeline   # no data needed
python scripts/mil_head.py                                          # needs torch
```

## Install

**Claude Code / Claude:** put the folder containing `SKILL.md` at `~/.claude/skills/mri-study-pipeline/`, or upload it as a skill.

**Codex:** put the folder at `~/.codex/skills/mri-study-pipeline/`, restart the session, and invoke it with `$mri-study-pipeline`.

## License

MIT, see `LICENSE`.
