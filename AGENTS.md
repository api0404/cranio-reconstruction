# AGENTS.md

## Project objective

Build a reproducible research pipeline for estimating craniofacial 3D geometry
from a lateral cephalometric radiograph, validating it against independent
geometry, and eventually transferring it to a high-resolution anatomical
model. This is exploratory work, not a diagnostic or surgical-planning tool.

## Current status and scope

Six scoped experiments are complete:

1. The published SC-DREG `04002` inference example runs and has been compared
   quantitatively with all five committed upstream outputs.
2. A project-side clinical-ceph canonicalization pipeline makes explicit
   128 x 128 candidates with invertible **coordinate** transforms.
3. A local input-domain sensitivity study ran the three existing candidates,
   eight small perturbations, and an unmasked control through the reproduced
   model. It measures compatibility and self-consistency, not 3D accuracy.
4. An optional ALI-CBCT probe ran nine published midline landmark agents on
   the 04002 and subject reconstructions using an explicit, hypothetical
   intensity mapping. Subject reprojection discrepancies were measured against
   curated 2D marks; they do not validate 3D anatomy.
5. An optional pretrained py-ceph 2D detector was compared on the original
   ceph against the curated manual marks. It produced separate unreviewed
   predictions and manual/automatic landmark-driven 128 x 128 candidates.
   Detection and canonicalization remain distinct, and no 3D optimization
   followed.
6. A controlled manual-versus-py-ceph `anchor_tighter` comparison ran both
   candidates through the same SC-DREG checkpoint and repeated nine-label
   ALI probes. It isolates input sensitivity; manual reconstruction is the
   experimental reference, not 3D anatomical ground truth.

For a new task, follow the user's requested scope. Do not automatically begin TrueDepth,
high-resolution reconstruction, Blender, GUI, osteotomies, fixation plates,
clinical deployment, or model retraining. Do not present the current 3D output
as patient-specific anatomical truth.

## Read first

- `docs/sc_dreg_notes.md`: verified upstream input, architecture, projection,
  deformation and output behavior; compatibility changes; published-example
  numerical comparison.
- `docs/clinical_ceph_canonicalization.md`: evidence and unknowns about
  clinical preprocessing, transform conventions, candidate generation.
- `docs/sc_dreg_input_sensitivity.md`: local ceph mask, 12-case experiment,
  metrics, visualization paths, recommendation and limitations.
- `docs/ali_cbct_probe.md`: optional ALI-CBCT detector source, landmark
  correspondence, intensity/geometry compatibility, experimental results.
- `docs/ceph_2d_landmarks.md`: 2D detector selection, compatibility,
  comparison against curated points, and landmark-driven preprocessing.
- `docs/manual_auto_sc_dreg_comparison.md`: paired manual/automatic
  reconstruction, transform, deformation, segmentation and ALI differences.
- `docs/experimental_ceph_alignment.md`: superseded exploratory alignment;
  its visual anchors are not reviewed cephalometric landmarks.

The public upstream implementation is pinned under `vendor/sc-dreg/` at
`8fd958ff195b3a4ebdc2bed9cc5bdb2c5fe25eb1` and remains unchanged.

## Verified handoff facts

- Upstream `read_img` converts an already prepared image to grayscale and a
  `[0,1]` tensor; `test.py` reshapes it to `[1,1,128,128]`. Upstream does not
  provide a clinical crop, spacing calibration or resize recipe.
- `tests/04002.png` is a 128 x 128 grayscale PNG without physical-geometry
  metadata. Whether it originated as a clinical image or synthetic DRR, and
  how it was prepared, remain unknown.
- On this AMD/CPU machine, the published refined volume had MAE `5.08239e-8`
  against the committed reference. Raw PNG and registered segmentation matched
  exactly; coarse and refined DRRs differed at 16 and 22 pixels, respectively.
  See the baseline audit for the discrepancy analysis.
- `coeff4.npy` is `(129, 6291456)` float64, about 6.05 GiB of array data.
  The runner memory-maps it and materializes only the first 60 rows as float32.
- The local private ceph is 2808 x 2136. The candidate-generation config
  contains rough visual anchors. A separate private landmark export now
  contains curated S/N/Ba/Me and other 2D cephalometric points. They permit
  2D reprojection discrepancy measurement, but no paired CBCT exists and
  patient-specific 3D accuracy cannot be measured.
- The sensitivity study found substantial dependence on FOV and even small
  translations. `tighter_fov` is the documented first exploratory candidate,
  not an established physical normalization. The vertical ruler body was
  masked in a copy; a horizontal bracket remains. NIfTI unit spacing is a
  serialization default, not millimeters.
- Native ALI-CBCT histogram correction turns a unit-range SC-DREG volume into
  all-zero int16 data. The optional probe uses an explicitly hypothetical
  intensity map and keeps ALI source and weights isolated. Its predictions
  cannot be assigned clinical accuracy or physical millimeters.
- S/N/Ba/ANS/PNS/A/B/Pog/Me agents completed on both reconstructed volumes.
  Subject 2D reprojection discrepancies range from 3.80 to 14.51 model pixels;
  the largest is ANS. See the ALI note and ignored overlays for exact values.
- The py-ceph detector supports 13 directly comparable skeletal/dental marks
  on this ceph. Full-image detection has 23.42 px mean, 23.64 px median and
  60.04 px max original-raster error against manual marks. Letterboxing fails
  badly; detector input hypotheses are not SC-DREG normalization rules.
- No independent clinical measurement table is present. SNA/SNB/ANB in the
  2D report are derived from the curated and predicted point coordinates.
- Replacing the five manual fit landmarks with py-ceph marks in
  `anchor_tighter` moves the crop center 10.59 source pixels, changes
  rotation 0.449 degrees and scale 1.268%. Refined-volume correlation is
  0.9932, mandibular Dice 0.9709. The largest 2D detector outliers Po and
  ANS were excluded from the fit. See the paired comparison for all metrics.

## Reproduce the completed work

Initialize the Python environment as documented in `README.md`, then place
the required untracked assets in `models/`. The published baseline uses:

```powershell
.venv\Scripts\python.exe scripts/run_sc_dreg_demo.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The local clinical-ceph experiments require the private image and ignored
configuration files already under `data/private/` on this machine:

```powershell
.venv\Scripts\python.exe scripts/canonicalize_ceph.py data/private/sc_dreg_sensitivity.json --output-dir outputs/sc_dreg_sensitivity/inputs
.venv\Scripts\python.exe scripts/study_sc_dreg_input_sensitivity.py
```

`scripts/study_sc_dreg_input_sensitivity.py` regenerates its candidate inputs
itself and saves `outputs/sc_dreg_sensitivity/study.json`, case-level PNG/NIfTI
files, overlays, montage and volume-difference views. These are private,
generated and ignored by Git. The input configuration is also private and
ignored; the exact ruler-mask polygon and assumptions are documented in the
sensitivity note. Do not copy that polygon to another patient's image.

The optional ALI experiment uses `scripts/ali_cbct_probe.py` and
`scripts/analyze_ali_cbct_probe.py`; see `docs/ali_cbct_probe.md` for the
separate environment, model downloads, commands and coordinate caveats.

The optional 2D detector uses `scripts/run_ceph_2d_detector.py` and the
landmark adapter `scripts/canonicalize_from_ceph_landmarks.py`; see
`docs/ceph_2d_landmarks.md` for the separate environment, exact commands,
case-specific experimental template and accuracy limits. Its ignored
`vendor/py-ceph/` checkout and pickle checkpoint must remain untracked.

The paired experiment runs `scripts/compare_manual_auto_sc_dreg.py` after
the two `anchor_tighter` inputs pass through `run_sc_dreg_demo.py`;
`docs/manual_auto_sc_dreg_comparison.md` gives exact commands and results.
Optional ALI runs on both volumes add projected landmark comparisons.

## Engineering and data rules

- Inspect source, model assets and measured runtime before changing behavior.
  Clearly separate **VERIFIED**, **INFERRED** and **UNKNOWN** statements.
- Keep `vendor/sc-dreg/` isolated. Prefer project-side compatibility wrappers;
  validate any change in numerical behavior against the `04002` reference.
- Preserve the original ceph. The 3 x 3 spatial transform maps coordinates
  both ways; raster crop/downsampling and clipped intensity values are not
  recoverable from the 128-pixel candidate. Do not imply otherwise.
- Use explicit orientation, rotation, translation, FOV and uniform scale.
  Do not introduce non-uniform scaling, shear or anatomical warping without
  direct evidence from the training pipeline.
- A higher input-to-DRR NCC/SSIM or a plausible-looking volume is only
  self-consistency. Do not select a clinical normalization or claim accuracy
  from these measures alone.
- Keep private cephs, DICOM/CBCT/NRRD files, landmarks and local configurations
  in `data/private/` or `data/raw/`; keep model assets in `models/` and generated
  medical outputs in `outputs/`. These paths are ignored. Never commit or
  upload private medical data without explicit user direction.
- Prefer small, testable project scripts and keep changes reviewable. Run the
  relevant tests, confirm ignored assets remain untracked, and leave Git clean.

## Useful next evidence, when authorized

Independent marks on model DRRs would help separate ALI errors from SC-DREG
reconstruction error. Original DICOM
detector spacing, source-to-detector geometry, or author-provided training
preprocessing would help resolve FOV and physical scale. A paired clinical
CBCT would be needed for 3D accuracy evaluation. Until such evidence exists,
carry forward the current uncertainty rather than inventing calibration.
