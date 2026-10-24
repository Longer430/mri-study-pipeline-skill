---
name: mri-study-pipeline
description: "Build the image side of a study-level MRI classifier: DICOM ordering, laterality, series typing, physical crops, anatomy-anchored ROIs, 2.5D attention-MIL models, ensembling and robust offline inference."
license: MIT
---

# MRI Study Pipeline

**When to use:** a multi-series MRI task with study-level labels (several findings per study, no slice-level labels), especially a Kaggle-style setup with offline, time-limited inference on a hidden test set. Typical: knee, spine, shoulder or brain MRI with 3–14 series per study.

**Related skill:** labels come first. If the training targets are report-derived or LLM-extracted, use `medical-report-label-refinement` for the label pipeline and validation discipline; this skill covers the image side.

**Origin:** the post-competition write-ups of the RSNA 2026 Knee Abnormality Detection Kaggle competition (21st, 22nd, 32nd, 33rd, 38th, 49th, 69th, 72nd, 73rd, 74th, 76th, 102nd, 220th, 246th places, YYama's ROI post, Tom's IC-Zoom post, and several bronze/no-medal write-ups with useful negative results). Report to the user in plain language (Chinese for this user).

**Main lesson:** the backbone barely matters once it is a reasonable 2.5D CNN; gains come from input geometry, looking at the right anatomy, and an inference run that never silently fails.

## 1. DICOM to arrays

- **Sort slices along the plane normal** (ImagePositionPatient projected on the slice normal), never by instance number or file name. One pipeline was capped at 0.65 AUC until slice order was fixed.
- **Decode defensively.** The training set may be uncompressed while the hidden test uses mixed transfer syntaxes. Bundle offline decoders (pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg, python-gdcm; verify wheel hashes) and wrap each file in try/except so a corrupt slice costs a slice, not a study.
- **Audit failing studies on the hidden run.** Even after decoding worked, one team found 1–13 hidden studies still failing preprocessing and falling back to a prior; fixing them was worth about +0.001.
- **Intensity:** per-series percentile scaling (1–99 or 0.5–99.5) to uint8; set the floor at background so dark structures keep contrast.
- **Preprocess next to the data** (e.g. Kaggle CPU sessions, sharded) into a compact cache; mounting hundreds of GB of small files remotely is very slow. Keep the original DICOM archive separate from model caches so you can re-crop or re-render later.
- **Each pretrained or public model keeps its own preprocessing.** Feeding one pipeline another pipeline's resized cache creates an input mismatch.
- Handle studies with two knees / two sides: process each side separately and combine per finding (e.g. max).

## 2. Laterality and orientation

- **Canonicalise every study to one side** (e.g. all left knees mirrored to right). Order: DICOM Laterality tag → image position → a small classifier on a few slices (≈99.5% accurate in reports); when the tag and the image disagree, trust the image.
- In knee MRI only the **sagittal** position carries the side; coronal/axial offsets do not. Test this on an audit sample before trusting it.
- Check for **reversed slice order** (sagittal) and **reversed in-plane orientation** (coronal/axial) separately; they can occur together in one study. One team trained a per-plane orientation head and combined plane decisions into one study decision.
- After canonicalisation, **do not use horizontal-flip augmentation or flip TTA**: it swaps compartments (−0.006 in one test). Use "anatomical mirror" TTA only if the model is trained for it with label swapping.

## 3. Series typing and selection

- **Do not trust SeriesDescription.** An audit of 24,371 series found names wrong whenever they conflicted with the header (re-scans keeping an FS name, Dixon reconstructions sharing a name, mislabelled sequences), and more than half had no usable name.
- Derive attributes from header fields: plane from orientation; obliquity from the largest slice-normal component (< 0.9 → oblique); fat suppression from flags, ScanOptions (FS/WE), inversion recovery, or vendor-specific rules (e.g. water-selective 3D GRE); weighting from sequence type and TR/TE (e.g. T1 when TE < 30 ms and TR < 1200 ms, else PD/T2); 2D/3D from header or slice thickness. Fall back to the name only when the header lacks TR/TE.
- Check host-provided flags for their real meaning (one host's fat-suppression flag meant "fluid is bright", so FS T1 series were flagged 0). Validate rules with an image classifier (e.g. FS from slice appearance) and turn its systematic misses into rules.
- **Selection:** drop exact repeats (keep the latest series number), require a minimum slice count, fill per-plane slots preferring fluid-sensitive series first and new types next, and let empty slots take leftovers. Or skip slots entirely and read all series (Section 5).
- **Do not feed site-identifying metadata** (sex, scanner model, coil, field strength) as inputs when labels carry site habits; it becomes a shortcut. Acquisition attributes that describe contrast (weighting, FS, TE bin) can be fed as zero-initialised token embeddings with random dropout (e.g. 15%) so the model does not lean on them.

## 4. Input geometry (spend time here early)

- **Crop by physical size, not pixels:** a central 130–160 mm field at about 0.3–0.4 mm/px (336–448 px) worked across teams. Going from 224 px to 336 px over 130 mm was one team's only single-change gain that cleared its adoption bar (+0.011 on the expert set); 448 px added nothing. 100 mm cut anatomy off.
- **2.5D windows from physical neighbours** (slice ± ~3.5 mm) as RGB, never crossing a series boundary.
- **Read all slices** where the budget allows. Fixed per-series slice counts dropped ~28% of slices in one pipeline, mostly from long series, hurting thin structures (menisci). Subsample very thin 3D series (e.g. > 60 slices to 32 levels) so they do not flood a bag.
- **Look at what the model sees.** If you cannot see a millimetre-scale finding at the chosen resolution and crop, neither can the model.

## 5. Study model

The common strong design:

1. A shared 2D backbone encodes each 2.5D window (ConvNeXt-T/S, CoAtNet rmlp-2, EfficientNetV2-S, NFNet-L0; ImageNet-22k or 12k weights).
2. Optional mixer: a 1–2 layer transformer over windows (across series), with plane / position / contrast embeddings.
3. **Per-finding attention-MIL pooling** (one attention head or learned query per finding, Query2Label style), so each finding picks its own slices; then a per-finding linear head.

Details that mattered:

- **Zero-initialise new residual modules** (window transformer, 3D RoPE, gates) so training starts from the plain pooling model; one team said training collapsed otherwise.
- **Keep spatial information.** Global average pooling erases which side a lesion is on. Options that helped: feature-map cells (~12 mm) as MIL instances; 2×2 quadrant pooling with avg + max + GeM (one team 0.941 → 0.954); anatomical Gaussian priors added to spatial and depth attention. Learned spatial attention left alone tended to become uniform.
- **Two-level MIL** (cells/windows within a series, then series within a study) handles any number of series without fixed slots.
- 3D patient-coordinate RoPE across series and explicit 3D splatting mostly moved accuracy between findings rather than raising the mean; treat them as diversity, not as the main lever.
- Training recipe that worked broadly: AdamW (backbone 1e-4, head 1e-3, wd 0.05), short warm-up + cosine, EMA or SWA over the last epochs, soft-target BCE with per-cell weights, gradient checkpointing, BF16/FP16, 12–40 epochs, sampled windows in training (12–32) and dense windows at inference.
- **Domain pretraining:** continuing the trunk on a labelled in-domain slice set (RadImageNet knee MR) helped one team (0.946 → 0.954); external expert-labelled cohorts (OAI MOAKS) helped as pretraining or a 50/50 blend member for some findings. Frozen foundation features, DINOv3, SigLIP and in-domain masked-image pretraining generally lost to a fine-tuned ConvNeXt.
- **Bigger backbones never won** at the same recipe across teams; neither did SAM, AUC losses, mixup, longer schedules or warm restarts (all within seed noise).

## 6. Look at the right anatomy

Anatomy-anchored crops were among the largest single gains once labels were good.

- **Anatomy ROI per series:** YYama had Claude Code subagents draw anatomical (not pathological) boxes on 16 slices per series using millimetre-margin rules, with human QC of a montage per series; 50 series per plane were enough to train YOLO11s (joint-box IoU median 0.81–0.87). Cropping the joint box per series gave LB 0.945 → 0.951. Derived boxes (joint-space band, medial/lateral halves) can be computed from approved boxes. Small structures (ligaments) were not reproducible as boxes; fluid and cysts were.
- **Landmarks from a segmentation:** a bone mask (e.g. TotalSegmentator femur, licence permitting) → condyles, notch, joint line, trochlea → fixed-size (e.g. 64 mm) crops warped from native slices. Never feed the mask itself.
- **Detectors for local views:** small detectors (RF-DETR, YOLO) trained on public anatomy sets or LLM-drawn boxes, cross-checked in 3D across planes.
- **Structural isolation for side-specific findings:** a compartment classifier that sees one compartment at a time, with shared weights and the other side mirrored, cannot infer lateral disease from medial disease. Only this structural fix worked for one gold team; post-hoc subtraction lost.
- **Zoom on weak findings:** reuse a model's attention to pick windows, re-read them at native resolution zoomed ~2×, and blend into only those columns. Redo the selection per fold to avoid leakage.
- Plain region crops without anchoring, or crop models used as standalone members, did not help in several teams.

## 7. Ensembling

- **Diversity that helps:** different label versions, different views of the data (whole-study vs anatomy crops, single-plane models), different supervision sources. Seeds and small recipe changes produce predictions more correlated than two seeds and add nothing.
- **Blend method:** per-finding rank (percentile) averaging is the common default; probability or standardised-logit averaging also worked. Keep weights simple; set per-finding weights on large OOF sets, never on a small expert set. Stackers lost to plain means.
- **Weak members hurt:** a member below ~0.94 alone usually lowered a 0.95 blend. Have one strong model of your own before blending.
- Choose final submissions as one simplest well-averaged variant plus one independent hedge; when the leaderboard cannot separate variants, the fourth decimal is noise.

## 8. Offline, time-limited inference (treat the hidden run as part of the model)

- **Measure the hidden-run time of the base notebook** (submission to score is an upper bound) and budget every new stage against it, not against the few visible test studies. Several teams' last submissions silently fell back to an earlier stage because new stages ran past their guards.
- **Each member in its own stage/process with a timeout;** renormalise blend weights over members that finished. One dead member once produced a constant 0.500 submission for two teams.
- **Canaries:** if a stage fails, set one column to a constant so the score visibly drops instead of looking unchanged.
- **Graceful degradation:** if projected time exceeds a threshold, switch remaining studies to fewer evenly spaced windows; under memory pressure, reduce windows instead of crashing; fp16 rows that overflow can be recomputed in fp32; per-case fallbacks and an atomic, validated CSV write.
- **Force every fallback once** on a test run before you need it.
- **Speed:** decode each study once and share it across all inputs; overlap loading of the next study with GPU compute (−21% overall in one team); split models across both GPUs; torch.compile; multi-threaded DICOM reading (~10× in one test); run segmentation/cropping in a background thread.
- **Do not add new code on the last day.** If you must, compare it against the original path on many studies.

## 9. Checklist before training the first serious model

1. Slices sorted by physical position; laterality canonicalised and audited; no flip augmentation.
2. Series attributes from headers, rules validated against an image classifier.
3. Physical crop and resolution chosen by looking at slices; ROI crop if anatomy is off-centre.
4. Folds grouped by patient and scanner fingerprint.
5. Offline decoders bundled; inference notebook with stage timeouts, canaries and fallbacks tested end to end.
6. Labels and validation set up per `medical-report-label-refinement`.

## 10. Tried by several teams, did not help (do not try these first)

Each item was reported independently by at least two RSNA knee teams. Treat it as a low prior, not a ban.

- **A bigger backbone at the same recipe** (49th, 72nd, 74th, 76th, Civitasmass): never won.
- **DINOv3, SigLIP or frozen foundation-model features** (49th, 76th, 507th, Civitasmass): worse than a fine-tuned ConvNeXt, or level at much higher cost.
- **Recipe tweaks on a working model: SAM, AUC losses, mixup, longer schedules, warm restarts, stronger or searched augmentation** (49th, 76th, 102nd, Civitasmass): inside seed noise. (Weight averaging such as SWA/EMA is part of several strong recipes; adding it as a late tweak did not move scores.)
- **In-domain self-supervised pretraining (SimMIM and similar)** (49th, 76th): did not beat ImageNet initialisation.
- **Horizontal-flip TTA after laterality canonicalisation** (49th, 76th, Civitasmass): hurts, because it swaps compartments.
- **Resolution above ~336–384 px at a fixed physical crop** (49th 448 px, Tumer 448 px, 38th 352 px rejected): no gain.
- **More seeds of one recipe, or full-data refits replacing 5 folds** (49th, 32nd, Civitasmass): invisible on the leaderboard.
- **Region crops without an anatomical anchor, used as standalone members** (102nd, 412th, 76th synovium crops): weaker alone, no blend gain.
- **Adding a weak member (below ~0.94 alone) to a 0.95 blend** (412th, Northstar, Tumer): lowers the score.
- **Stacking, or blend weights fitted on a small expert set** (49th, 412th): lost to plain means.

## 11. Files in this skill

If only `SKILL.md` is present (for example a copy saved directly into Claude), the other files are in the GitHub repository Longer430/mri-study-pipeline-skill and in the user's Codex copy (`~/.codex/skills/mri-study-pipeline/`).

- `references/rsna-knee-writeups.md`: per-team digest of the image-side write-ups (inputs, crops, models, inference engineering, reported numbers, links). Read it for details behind a rule above.
- `scripts/dicom_series.py`: Sections 1–3. Slice ordering along the plane normal (fixed anatomical direction), header-based series typing (plane, oblique, fat suppression, weighting, 2D/3D, name only as fallback) and laterality (tag, then sagittal x sign, else undetermined). Works on pydicom Datasets or dicts; CLI prints one row per series of a study folder.
- `scripts/mil_head.py`: Section 5. Zero-initialised window mixer, per-finding attention-MIL pooling, zero-init tag embeddings with tag dropout, prior-logit init and masked soft-target BCE (PyTorch). `python scripts/mil_head.py` runs a shape test.
- `scripts/staged_inference.py`: Section 8. Runs each member as a subprocess with timeout and budget check, validates outputs, rank-blends survivors with renormalised weights, sets a canary column when a member failed, writes the CSV atomically, prints a JSON log.
- `scripts/selftest_pipeline.py`: synthetic checks for `dicom_series.py` and `staged_inference.py`. Run it first.

Dependencies are in `requirements.txt` (pydicom only for the DICOM CLI, torch only for `mil_head.py`).

---

Copyright (c) 2026 Wenting Long ([@Longer430](https://github.com/Longer430)). Released under the MIT License.