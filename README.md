# Cranio Reconstruction

A research and fabrication pipeline for creating an anatomically plausible,
patient-approximated 3D craniofacial model from a lateral cephalometric
radiograph.

The immediate goal is to reproduce and validate SC-DREG, then use its estimated
3D craniofacial geometry as the starting point for a high-resolution,
printable skull model.

The long-term model is intended for visualization and physical demonstration of:

- Le Fort I osteotomy
- bilateral sagittal split osteotomy (BSSO)
- sliding genioplasty
- modular postoperative segment positions
- interchangeable fixation plates for different movements

This is an experimental visualization / educational project. It is not a
substitute for diagnostic CBCT, surgical planning, or patient-specific clinical
decision-making.

## Pipeline

```text
High-resolution lateral cephalogram
                |
                |  normalized input
                v
            SC-DREG
                |
                |  estimated 3D deformation / CBCT
                v
      Craniofacial reconstruction
                |
                +----> cephalometric validation
                |          |
                |          v
                |    target-vs-estimate error
                |
                v
     high-resolution skull template
                |
                |  deformation transfer
                v
       refined craniofacial mesh
                |
                v
             Blender
                |
        +-------+--------+
        |       |        |
      LF1     BSSO     Genio
        |       |        |
        +-------+--------+
                |
                v
      modular fixation system
                |
                v
          printable models
```

## Current research status

The published SC-DREG `04002` example is reproduced quantitatively: the
refined volume has MAE `5.08e-8` against its committed reference, with exact
raw PNG and segmentation matches. See [the baseline audit](docs/sc_dreg_notes.md).
The [clinical ceph canonicalization](docs/clinical_ceph_canonicalization.md)
and [input-domain sensitivity study](docs/sc_dreg_input_sensitivity.md) test
explicit 128×128 candidates without assigning physical calibration.

An [optional ALI-CBCT compatibility probe](docs/ali_cbct_probe.md) tests
whether a published 3D CBCT landmark detector can recognize locations in the
synthetic SC-DREG output. Native ALI preprocessing erases SC-DREG's unit-range
intensities, so this probe uses a documented hypothetical intensity mapping.
Its landmarks are experimental annotations, not patient-specific 3D truth.

An [optional pretrained 2D landmark probe](docs/ceph_2d_landmarks.md) compares
py-ceph predictions on the original clinical image with the curated manual
marks, then generates separate manual- or automatic-landmark-driven 128 × 128
SC-DREG candidates. The manual export remains the reference and the automatic
predictions require review.

A [controlled manual-versus-automatic comparison](docs/manual_auto_sc_dreg_comparison.md)
passes both `anchor_tighter` candidates through the same SC-DREG checkpoint.
The two refined volumes correlate at 0.9932 and their registered mandibular
masks have Dice 0.9709. These are input-sensitivity results, not anatomical
accuracy measurements.

The [reference and deformation exploration](docs/sc_dreg_reference_deformation.md)
exports model-grid reference meshes, captures coarse and refined fields, and
summarizes all 60 used PCA directions. Its current subject run uses an
explicitly approximate unmasked input because the prior private candidate is
unavailable in this checkout.

The [subject skull inspection export](docs/subject_skull_inspection.md)
recreates a masked, manual-informed input from the prior rounded transform,
runs SC-DREG and exports local PLY skull and warped-reference-mandible meshes
with multi-view previews and an intensity threshold sweep.

## Repository layout

```text
cranio-reconstruction/
├── AGENTS.md
├── README.md
├── configs/
│   └── default.yaml
├── data/
│   ├── examples/       # small public/test inputs
│   ├── raw/            # local source data; ignored
│   ├── processed/      # generated intermediates; ignored
│   └── private/        # private medical data; ignored
├── docs/
│   ├── architecture/
│   ├── research/
│   └── sc_dreg_notes.md
├── models/             # pretrained assets; ignored
├── outputs/            # generated reconstructions; ignored
├── artifacts/          # large generated deliverables; ignored
├── scripts/
├── slicer/
│   └── scripts/
├── blender/
│   ├── assets/
│   └── scripts/
├── src/
│   ├── ceph/
│   ├── geometry/
│   ├── landmarks/
│   ├── mesh/
│   ├── reconstruction/
│   └── utils/
├── tests/
└── vendor/
    └── sc-dreg/        # upstream Git submodule
```

## Upstream model

SC-DREG:

https://github.com/Jyk-122/SC-DREG

Paper:

> Towards Semantically-Consistent Deformable 2D-3D Registration for 3D
> Craniofacial Structure Estimation from A Single-View Lateral Cephalometric
> Radiograph

The upstream implementation predicts PCA deformation parameters from a
128 x 128 lateral cephalogram and refines a 128 x 128 x 128 deformation field.

The upstream source remains isolated under `vendor/sc-dreg/` so that project
code can evolve without losing a clean reference implementation.

## Required pretrained assets

Place these files manually in `models/`:

```text
cbct_c2f_model_ckpt.tar
coeff4.npy
mean4.npy
ref.nii.gz
seg_mandible.nii.gz
xray_seg_unet_ckpt.tar
```

These files are intentionally excluded from Git.

The large `coeff4.npy` should initially be inspected with NumPy memory mapping
rather than loaded fully into memory.

Run:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-sc-dreg.txt
.venv\Scripts\python scripts/inspect_model_assets.py
.venv\Scripts\python scripts/check_sc_dreg_checkpoint.py
.venv\Scripts\python scripts/check_sc_dreg_reference_drr.py
.venv\Scripts\python -m unittest discover -s tests
.venv\Scripts\python scripts/run_sc_dreg_demo.py
```

The demo writes to `outputs/sc_dreg_demo/` and compares all five outputs
numerically with the upstream example. It reports missing assets before
inference. See [the baseline audit](docs/sc_dreg_notes.md) for observed shapes,
compatibility details and open input assumptions.

## Data policy

Private medical imaging data belongs only in:

```text
data/private/
```

Raw source volumes belong in:

```text
data/raw/
```

Both directories are excluded from version control.

Generated NIfTI, NRRD, model checkpoints, raw CBCT/DICOM files, and large mesh
artifacts should not be committed to the repository.

## Git workflow

The root project is its own Git repository.

The upstream SC-DREG implementation is tracked as a Git submodule, preserving
its original history and preventing local modernization work from becoming
mixed with upstream source.

Suggested branch naming:

```text
main
research/sc-dreg-reproduction
feature/ceph-preprocessing
feature/landmark-validation
feature/highres-deformation
feature/blender-osteotomies
feature/modular-plates
```

Keep commits focused and make compatibility changes separately from behavioral
changes.

Before any model modernization, establish a reproducible reference output from
the original example.

## Planned stages

### Stage 1 — SC-DREG reproduction

- reproduce upstream example inference
- inspect pretrained assets
- document projection geometry
- modernize runtime only where necessary
- numerically compare outputs

### Stage 2 — Clinical cephalogram preprocessing

- preserve original high-resolution cephalogram
- determine crop, orientation, scale and projection assumptions
- generate model-compatible 128 x 128 input
- retain physical calibration independently

### Stage 3 — Reconstruction validation

- detect or annotate cephalometric landmarks
- compare reconstructed and source cephalometric measurements
- quantify sagittal and vertical errors
- distinguish observed geometry from statistical 3D estimates

### Stage 4 — High-resolution geometry

- transfer the estimated deformation field to a higher-resolution anatomical
  template
- preserve fine anatomy that SC-DREG cannot recover at 128^3
- generate a clean printable skull mesh

### Stage 5 — Orthognathic model

- Le Fort I
- BSSO
- sliding genioplasty
- independent bone segments
- parameterized postoperative movements

### Stage 6 — Modular fixation

- interchangeable plate sets
- real or miniature metal screws
- repeatable movement-specific segment positioning
- printable demonstration system

## Important limitations

A single lateral cephalogram cannot uniquely determine full 3D anatomy.

In particular, transverse dimensions, asymmetry, detailed condylar morphology,
root geometry, mandibular canal anatomy, cortical thickness and other
out-of-plane structures are estimated from the statistical prior rather than
directly observed.

The resulting skull is therefore an approximation constrained by the available
2D anatomy, not a replacement for a real CBCT.
