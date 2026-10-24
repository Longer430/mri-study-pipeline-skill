# RSNA 2026 Knee: what top write-ups did on the image side

Digest of the post-competition write-ups on the [discussion forum](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion), first two pages, read 2026-10-23. Summaries are in my own words; numbers are as the authors reported them. Label-side details are in the `medical-report-label-refinement` skill's digest.

Setting: multi-series knee MRI (5.5 series per study on average, 24,371 series, 819,078 slices, about 570 GB of DICOM), 12 study-level findings, macro AUC, offline inference on 2×T4 within 9 hours on about 1,300 hidden studies with mixed DICOM transfer syntaxes.

## Per-team digest

### 21st (solo gold) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/21st-place-solo-gold-solution-llm-soft-labels)

- Multiview 2.5D ConvNeXt-Small from a public notebook (heliosli): centre slice ± ~3.5 mm as RGB; full-field plus local views (sagittal/axial central, patellofemoral, posterior; coronal both sides); class-specific spatial attention plus attention across windows.
- 384 px cache, random 352 crops, up to 176 windows per study in training, denser at inference; AdamW 1e-4 / 1e-3, wd 0.05, 5% warm-up, cosine; one study per step with 2-step accumulation; no flips; early stopping on fold macro AUC.
- Final: 45% LLM-label models (continued 2 epochs at lower LR), 45% self-distilled retrain, 10% a public model; fixed weights, no rank transforms.
- Window cap 176 → 192, position-stratified sampling, light gamma augmentation: all within paired CIs. A 3-crop inference submission timed out on the hidden set.

### 22nd (solo gold, private 0.954) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/22nd-place-solo-gold-solution)

- Preprocessing: slices sorted along the plane normal, 1st–99th percentile to uint8, central 140 mm → 336 px (384 at input), all knees to left orientation (Laterality tag → image position → ResNet18 on coronal slices; image wins on conflict). Two-knee studies: each knee separately, max per finding.
- Series audit of all 24,371 series: names wrong whenever they conflicted with the header; more than half had no usable name. Rules: plane from host table; oblique if largest normal component < 0.9; FS from host flag, ScanOptions FS/WE, inversion recovery, or Philips 3D GRE with TR ≥ 10 ms; weighting ZTE if TE < 1 ms, STIR if IR, GRE if gradient echo, else T1 if TE < 30 ms and TR < 1200 ms (800 for 3D), else PD-T2 with TE passed as a number. A slice-appearance FS classifier agreed with the rules at AUC 0.9994.
- Series selection: slots 3 sagittal / 3 coronal / 2 axial, first slot prefers FS, later slots prefer new types; slice counts 18/12/12, 14/8/8, 8 over the central 88%; at most 8 series and 88 slices.
- Femur segmentation (TotalSegmentator MRI femur class) → condyles, notch, joint line, trochlea → 64 mm crops warped to 336 px (~0.19 mm/px); the mask is never fed to a model.
- Models: 2.5D window encoder + per-finding attention pooling (score w·tanh(Vh), softmax over windows); 2-layer 8-head window transformer with zero-initialised output added as a residual (without it training collapsed); zero-initialised embeddings for plane, position and acquisition attributes; weighting attribute hidden with p = 0.15.
- Compartment classifier with shared medial/lateral weights (lateral mirrored), so it cannot see the other compartment; only this structural decoupling worked (post-hoc subtraction −0.007 public). Lateral OA weight 74% from this model.
- Final: 56 networks in 12 groups (CoAtNet rmlp-2 and CoAtNet-3 main models, ConvNeXt-T single-plane models, compartment/ACL/joint-ROI crop models), per-finding probability weights by hill climbing on 4,342-study OOF. Single changes to head/loss/backbone stayed more correlated with the baseline (0.955–0.97) than two seeds (0.93) and did not move the ensemble.
- Inference on 2×T4: decode each study once, segmentation and cropping in a background thread, GPUs split the models; 56 networks fit, 61 did not; a failed model is dropped and weights renormalised.

### 32nd (private 0.952) — [post](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751930)

- Model A: CoAtNet rmlp-2 attention-MIL over 5 slots, 140 mm → 384 px, truly adjacent DICOM slices as RGB, 54 windows at inference; evidence loss pulling attention to VLM-cited slices; learnable window position embedding; OAI MOAKS training variant averaged 50/50; scanner metadata residual; mirror TTA; SWA.
- Model B: ConvNeXt-T with slot and slice-position embeddings and hierarchical per-finding attention; OAI MOAKS pretraining then fine-tune.
- Lateral zoom model for lateral meniscus, lateral OA, PF OA: pick 6 window groups per finding from model A's attention (re-selected per fold), re-read at native resolution ~2× zoom; 20% rank weight on those three columns only.
- Speed: next-study loading overlapped with GPU compute (−21%), images moved to GPU once, torch.compile; time guard per stage.
- Public LB ladder: adjacent DICOM slices 0.932 → 0.943; image teacher + evidence loss 0.956 → 0.960; Model B 0.963; zoom 0.964.

### 33rd (private 0.952) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/33rd-place-silver-solution-no-external-data)

- All series and all slices, no selection; central 140 mm → 336 px (384 on GPU); per-series 2–98 percentile; window = centre ± 3.5 mm; 4 random windows per series in training, every window (~140) at inference, no TTA.
- Two-level attention MIL: feature-map cells (~12 mm) of every window in a series → per-finding series vectors → attention over series. Any number of series (3–14).
- 3D RoPE across series from DICOM patient coordinates (wavelengths 4–256 mm), zero-init output: +0.002 public / +0.001 private, within noise.
- OrthoFoundation-L (ViT-L/16 DINOv3 further trained on knees, last 8 blocks trained) vs CoAtNet rmlp-2: on par (0.962/0.947 vs 0.961/0.948) at ~4× the FLOPs. Patch embedding initialised to see only the centre slice; neighbour weights stayed at ~2% of the centre. Keep DINOv3's own 2D RoPE enabled with checkpoint periods. fp16 rows that overflow are recomputed in fp32.

### 38th / The Hospital Is a Variable Too (private 0.952) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/the-hospital-is-a-variable-too)

- Leg A: whole-knee 2.5D over 5 MRI slots (EfficientNetV2-S / NFNet). Leg B: landmark-anchored crops (medial, lateral, PF, posterior) at a fixed 3.5 mm step with per-class attention MIL (ConvNeXt-S, ConvNeXt-B CLIP), built on heliosli's public trainer.
- The stronger compact model carried less hospital signal (eta² 0.058 vs 0.075).
- Rejected under a gate (P(gain) ≥ 0.85, ≤ 5 candidates per round): 352 px, extra zoom views, V2-M, per-class blend weights, flip TTA.
- Fail-soft blend: each leg in its own process with a timeout, checked before entering the blend, weights renormalised.

### 49th (solo, private 0.951) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/49th-place-silver-solo-whole-study-2-5d-reader), [code](https://github.com/Komil-parmar/rsna-knee-2026-49th-place)

- Preprocessing in Kaggle CPU sessions into shards: decode, sort, left-knee convention, 160 mm at 512 px uint8; training cache 336 px over 130 mm for all 819,078 slices; 3D series > 60 slices subsampled to 32 levels. Laterality from the sagittal x-position sign where the tag is missing (coronal/axial offsets do not carry side).
- Folds by scanner fingerprint (manufacturer, model, imaging frequency to the MHz, coil; 83 groups): random folds inflated CV by 0.05–0.07; a metadata-only model scored 0.65 on random folds, 0.60 grouped.
- Slice-bag reader: 32 triplets per study from any series, ConvNeXt-Tiny/V2-Tiny tokens (384-d) with plane/contrast/position embeddings, 2-layer 6-head transformer, 12 learned finding queries with cross-attention; whole-study deterministic read at test time (96–160 slices per series).
- Input geometry was the only single change that cleared the adoption bar: 224 → 336 px over 130 mm +0.011 gold58; 448 px nothing. A week of 11 recipe lever families at 224 px all tied.
- Flip TTA −0.006 on the board. Frozen DINOv3-H+ decoder ceiling 0.826 vs 0.907 for a trained small model's features; SimMIM −0.008; bigger backbones never won.
- Test-time hardening: bundled JPEG-lossless/JPEG 2000 decoders, per-file try/except, each blend member in its own stage with a time guard; one dead member once produced a constant submission. Final notebook ran in about 7 of 9 hours.

### 73rd KneeSplatNet (private 0.949) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/kneesplatnet)

- Canonical knee frame from DICOM geometry (origin at the overlap of sagittal and coronal FS volumes; x lateral, y anterior, z superior); per-finding anatomical-anchor pooling plus a 12×12×12 trilinear splat grid read by 3D ConvNeXt blocks.
- Knockouts on the same checkpoint: Path B off −0.0049, geometry off −0.0011. It did not beat its 2.5D control overall but shifted accuracy toward PF OA and away from ACL, menisci, contusion.

### 74th (private 0.949) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/74th-place-knee-centred-views-llm-labels-and-oa)

- Five series (sagittal and coronal with/without FS, axial FS), every slice, central 140 mm at 448 px (~0.31 mm/px), per-series window with floor at background.
- Two RF-DETR Small detectors: meniscus (trained on a public CC BY coronal set, IoU 0.82 vs 0.75 for YOLO) and patella (boxes drawn by an LLM on 480 slices, kept only when sagittal, axial and a TotalSegmentator mask agreed; mAP 0.62). Six 100 mm local views plus full field.
- Inference guards: if projected time passes 7.5 h, remaining studies use 192 evenly spaced windows (−0.0002 OOF); under RAM pressure score a study with fewer windows.
- Members chosen on external expert sets never trained on (fastMRI+, MRNet, held-out OAI), not on gold58.

### 76th (private 0.948) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/rsna-knee-ai-challenge-solution)

- Up to six series of 24 slices, slot routing by plane and fluid-sensitive flag, left knees to right convention (tag on 2,178 studies, a 99.54% CNN elsewhere), 2.5D at 320 px, ConvNeXt-Small, 2-layer slice transformer per series, Query2Label head over all slice tokens.
- RadImageNet knee MR continued pretraining (179,555 labelled slices, head discarded): public 0.946 → 0.954. Canonicalisation + slice positions + query head + EMA: fold validation 0.876 → 0.895.
- Up to 60 checkpoints rank-averaged under a 6-hour budget (at least 30 passes, fold 0 of every family first); 16-thread DICOM reader ~10× faster.

### 102nd (private 0.947) — [writeup](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/102nd-place-local-llm-report-labeling-and-anatomy)

- 160 mm field at 512 px; fluid-sensitive series resampled to 24 slices over a fixed depth (80 mm sagittal/axial, 64 mm coronal).
- Stage 1: per-plane ResNet34 + U-Net heatmap for ROI centre plus an orientation head, trained on 250 hand-annotated studies; mean centre error ~1.3 mm, plane orientation accuracy 97.9%. Separate handling of in-plane reflection and slice-order reversal; out-of-image crops padded, not shifted.
- Stage 2: independent sagittal/coronal/axial experts; spatial weight = softmax(content score + 0.5 × log Gaussian prior per plane–finding); Gaussian-initialised depth bias; soft top-k (k = 3) fused by a learnable gate; manual plane priorities 3 : 2 : 0.5 per finding.

### YYama (private 30th) — [post](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751903), [notebook](https://www.kaggle.com/code/yosukeyama/knee-mri-anatomical-roi-yolo11-from-claude-labels)

- Joint often off-centre (coronal median 15 mm); a fixed centre crop cut >5% of the joint in 45% of coronal series.
- Coding-agent subagents drew anatomical boxes on 16 native slices per series with a pixel grid, using millimetre margin rules (e.g. sagittal joint box 60 mm above / 45 mm below the joint line, 20 mm behind the condyles), with confidence and keep/discard per slice; human QC per series montage; 50 series per plane.
- YOLO11s per plane: mAP50 0.94/0.82/0.95 (cor/sag/ax), joint-box IoU median 0.87/0.81/0.84; series box = per-coordinate median of slice boxes. 320 × 320 crop (~105 mm) of the joint box: LB 0.945 → 0.951.
- Pathology boxes on 200 studies: Baker's AUC 0.989, effusion 0.964, synovitis 0.691. Ligament boxes not reproducible as axis-aligned boxes.

### Tom: IC-Zoom — [post](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751895)

- Learned non-uniform zoom: saliency map → Gaussian attraction warp (saliency sampler) composed over 3 steps on the original slice, conditioned on region × view via FiLM, zero-initialised to identity, guided by a KL term toward rough YOLO boxes. Linearised multi-sampling gave usable gradients to the grid.
- Learned-mixin pair loss (stop-gradient partner logit) for lateral OA ← medial OA, PF OA ← medial OA, lateral meniscus ← ACL, synovitis ← effusion.
- Did not beat the fixed-crop model alone; useful as a diverse ensemble member.

### Bronze / no-medal write-ups with useful negative results

- [Northstar](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/public-0-958-private-0-942-what-65-submissions): public-notebook recipe dropped ~0.016 public → private vs ~0.010 for strong own models; 22 attempts to blend an own 0.937 model added at most +0.001; 3 of the last 5 submissions stopped on their own time guard.
- [412th](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/412th-place-bronze-per-finding-rank-blend-of-pu): new stages passed the 3-study check but likely overran on the full hidden set and fell back silently; every member below ~0.94 lowered a 0.95 blend.
- [Tumer](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751999): two hidden-run crashes with no log while a 1,300-study rehearsal ran clean; fixed with finite-logit checks per fold, fp32 retry in chunks, per-case model fallback, prevalence fallback and atomic validated CSV writes, each forced once in a 100-study test.
- [246th](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/246th-place-silver-report-labels-are-not-expert): pinned image could not decode JPEG Lossless (one model zero-filled silently, another dropped the series); sha256-checked decoder wheels; canary columns showed failed hidden reruns at 0.916.
