# SC-DREG fixed reference and deformation exploration

Status: completed exploratory audit, 2026-10-02. The public upstream checkout
is still pinned at `8fd958ff195b3a4ebdc2bed9cc5bdb2c5fe25eb1` and has no
project edits. The fixed reference, both predicted fields and all 60 used PCA
directions are inspectable under ignored `outputs/`.

## Reproduce

Use the model assets described in the README. On this machine, Python 3.12.6,
PyTorch 2.14.1+cpu, NumPy 2.4.6, SimpleITK 2.5.6, scikit-image 0.26.0 and
matplotlib 3.11.2 were used. Install `requirements-exploration.txt` in `.venv`.

```powershell
.venv\Scripts\python.exe scripts/explore_sc_dreg_reference.py
.venv\Scripts\python.exe scripts/run_sc_dreg_demo.py --capture-deformation
.venv\Scripts\python.exe scripts/prepare_documented_subject_approx.py
.venv\Scripts\python.exe scripts/run_sc_dreg_demo.py --input outputs/documented_subject_approx/subject_approx_unmasked_128.png --output-dir outputs/documented_subject_approx/inference --skip-reference-comparison --capture-deformation
.venv\Scripts\python.exe scripts/explore_sc_dreg_deformation.py --subject outputs/documented_subject_approx/inference/subject_approx_unmasked_128_deformation.npz
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For an exact prior subject comparison, replace `--subject` with a deformation
capture from the restored manual `anchor_tighter_128.png`, made with
`run_sc_dreg_demo.py --capture-deformation`. The old ignored subject candidate,
private transform and ruler-mask config were absent in this checkout. The
current approximation uses the **rounded** reported center `(1799.64,
1099.68)` source pixels, rotation `1.0502°` and FOV width `1603.57` source
pixels, with identity intensity and no ruler mask. It is a new unmasked input,
not an exact replication of the earlier manual reconstruction. Its generated
config and transform report are kept beside it.

## VERIFIED: fixed anatomy

`ref.nii.gz` and `seg_mandible.nii.gz` have 128³ float32 voxels, identity
direction, zero origin and unit header spacing. These header defaults **do not
calibrate millimeters or anatomical orientation**. NumPy arrays are `[z,y,x]`;
PLY vertex coordinates are `(x,y,z)` model indices. The reference intensity
range is `[0,1]`, mean `0.16512`, median `0`, 75th percentile `0.34696`, 95th
percentile `0.57435`; it is not CT HU. The supplied mandible is binary, with
23,052 nonzero voxels. Its 0.5 isosurface has 13,883 vertices and 27,794
triangles, bounded by `(22.5,36.5,14.5)` to `(107.5,90.5,78.5)` in `(x,y,z)`.

`outputs/reference_exploration/reference_mandible.ply` and three
`reference_intensity_*.ply` files are binary PLY meshes that open in Blender
or 3D Slicer. A deliberately small intensity sweep gives:

| Unit-range isovalue | Voxels at or above | Triangles |
| ---: | ---: | ---: |
| 0.35 | 512,331 | 721,894 |
| 0.55 | 113,825 | 276,594 |
| 0.75 | 65,287 | 205,380 |

These are visual candidates, not bone segmentations or known HU thresholds.
The 0.35 candidate includes a wide low-intensity structure and intersects
the model-grid boundary. Marching cubes does not repair or close boundary
surfaces. The JSON report includes header geometry, intensity histogram,
quantiles, mesh sizes and bounds. The original volumes are never modified.

## VERIFIED: field semantics and predicted displacement

Upstream `model.py` computes
`coarse = para @ COEFF[:60] + mean_`, then reshapes to
`[z,y,x,(dx,dy,dz)]`. `grid_sample` samples the fixed reference at grid plus
the displacement. RefineNet consumes the coarse field and projected image
features and predicts a **complete replacement field**. Upstream samples the
same reference at grid plus that field; it does not add it to coarse. The
reported `refined_minus_coarse` is an analytical difference, not an upstream
composition step. All vector magnitudes are **model-grid index units, not mm**.

| Case | Coarse mean / p95 / max | Refined mean / p95 / max | Change mean / p95 / max |
| --- | --- | --- | --- |
| Published 04002 | 1.297 / 4.236 / 8.032 | 1.314 / 4.315 / 8.688 | 0.095 / 0.380 / 4.623 |
| Approximate unmasked subject | 1.325 / 4.187 / 9.530 | 1.337 / 4.231 / 10.450 | 0.119 / 0.417 / 5.439 |

Over the **supplied reference mandible mask**, the mean coarse/refined/change
magnitudes are `4.393/4.530/0.524` for 04002 and `5.352/5.399/0.494` for
the approximate subject. This mask is a fixed grid region, not a validated
subject landmark or registered anatomical region. The largest refined
displacement occurs at `[z,y,x]=[54,103,76]` for 04002 and `[17,39,64]`
for the approximate subject. The largest refinement change occurs at
`[114,37,77]` and `[50,59,82]`, respectively. Whole-field magnitude maps,
plane vector plots, 3D subsampled vectors and top 1% bounding boxes are in
`outputs/deformation_exploration/` and `analysis.json`.

The saved coarse fields reconstruct from the saved parameters and the first
60 float32 basis rows with maximum absolute component difference
`2.86e-6` in each case. That is consistent with floating-point matrix
multiplication order. The 04002 rerun's refined volume MAE against the
committed upstream reference is `5.07e-8` and its rounded segmentation is
exact. This checks the capture path did not alter model behavior.

## VERIFIED: PCA basis and mode ranking

`coeff4.npy` has 129 rows of 6,291,456 float64 values; only the first 60
are used. `mean4.npy` has one 6,291,456-value float64 displacement. Upstream
casts them to float32. A unit coefficient perturbation in mode `i` changes
the coarse field by row `i` reshaped to `[128,128,128,3]`. The script
memory-maps the basis and processes one row at a time, recording magnitude
mean, median, p95, p99, max and RMS over the whole grid and the reference
mandible mask, plus the maximum location and bounding box of the top 1% grid
magnitudes. `analysis.json` holds all 60 summaries and both predicted
coefficients. It ranks each case by `abs(coefficient) × basis RMS`.

| Zero-based mode | Unit RMS | 04002 coefficient / term RMS | Approx. subject coefficient / term RMS |
| ---: | ---: | ---: | ---: |
| 4 | 0.444 | 2.447 / 1.086 | 2.352 / 1.044 |
| 0 | 0.954 | 0.944 / 0.901 | 0.178 / 0.170 |
| 2 | 0.585 | -1.309 / 0.765 | -0.922 / 0.539 |
| 8 | 0.296 | 1.307 / 0.387 | 2.127 / 0.630 |
| 5 | 0.377 | -0.045 / 0.017 | -1.016 / 0.383 |

The top six global modes are `4,0,2,3,8,1` for 04002 and `4,8,2,5,1,6`
for the approximate subject. The union of these modes has unit-magnitude
projection maps, in-plane vectors at three z levels and 3D subsampled vector
plots under `outputs/deformation_exploration/modes/`. Each mode plot shows
grid displacement produced by **one unit coefficient**, not a standard
deviation. The rows need not be orthogonal in the metric used here, so these
individual term RMS values are not additive shares of total deformation.
Set `--top 60` to render every mode if a particular lower-ranked mode needs
visual inspection.

## UNKNOWN and interpretation limit

The public inference code and assets do not establish whether predicted PCA
coefficients are standardized, what training-space variance corresponds to a
unit coefficient, or the original physical voxel spacing and orientation.
No anatomical labels beyond the supplied mandible mask are assigned to PCA
effects. The approximate subject rankings could change when the exact prior
masked input is restored. None of the meshes or displacement measurements
validate patient-specific anatomy, bone threshold, clinical scale or accuracy.
