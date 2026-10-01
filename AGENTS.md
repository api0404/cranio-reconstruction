# AGENTS.md

## Project objective

Build a reproducible research pipeline for estimating craniofacial 3D geometry
from a lateral cephalometric radiograph, validating it against independent
geometry, and eventually transferring it to a high-resolution anatomical
model. This is exploratory work, not a diagnostic or surgical-planning tool.

## Current status and scope

Three milestones are complete:

1. The published SC-DREG `04002` inference example runs and has been compared
   quantitatively with all five committed upstream outputs.
2. A project-side clinical-ceph canonicalization pipeline makes explicit
   128 x 128 candidates with invertible **coordinate** transforms.
3. A local input-domain sensitivity study ran the three existing candidates,
   eight small perturbations, and an unmasked control through the reproduced
   model. It measures compatibility and self-consistency, not 3D accuracy.

The next milestone has **not** been authorized or defined. For a new task,
follow the user's requested scope. Do not automatically begin TrueDepth,
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
- The local private ceph is 2808 x 2136. Its current landmark file contains
  rough visual regions, **not** reviewed S/N/Ba/Me points. No paired CBCT or
  independent DRR landmark labels exist for it, so 2D reprojection error and
  patient-specific 3D accuracy cannot currently be measured.
- The sensitivity study found substantial dependence on FOV and even small
  translations. `tighter_fov` is the documented first exploratory candidate,
  not an established physical normalization. The vertical ruler body was
  masked in a copy; a horizontal bracket remains. NIfTI unit spacing is a
  serialization default, not millimeters.

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

Reviewed S/N/Ba/Me landmarks on the original ceph and independent marks on
model DRRs would permit a real 2D reprojection analysis. Original DICOM
detector spacing, source-to-detector geometry, or author-provided training
preprocessing would help resolve FOV and physical scale. A paired clinical
CBCT would be needed for 3D accuracy evaluation. Until such evidence exists,
carry forward the current uncertainty rather than inventing calibration.
