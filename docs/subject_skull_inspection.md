# Subject SC-DREG output: visual inspection and mismatch audit

Status: **unvalidated model estimate; visible mandibular mismatch**. The user
observed differences in gonial angle, Gonion height, and FMA between the ceph
and the reconstruction. These differences are anatomically material. Do not
use the exported mandible or skull as patient-specific geometry.

## Correction to the earlier export

The first inspection export mistakenly used the later manual `anchor_tighter`
crop: FOV `1603.57` source pixels. The project recommendation was the masked
`tighter_fov` template: FOV `1844.1` source pixels. The corrected local input
is `outputs/subject_inspection/recommended_input/manual_masked_tighter_fov_recreated_128.png`.
It uses the same original ceph, the documented same-image vertical-ruler mask,
identity intensity mapping, and a similarity transform reconstructed from
rounded report values: center `(1799.64,1099.68)` source pixels, rotation
`1.0502°`, FOV `1844.1` source pixels. The curated manual S/N/PNS/U1_tip/Me
marks are mapped as checks. In the earlier experiment a manual fit recovered
the `tighter_fov` template; its exact private transform/input files are absent,
so this is a near recreation, not a byte-exact rerun. The generated config,
source hash and 3×3 coordinate transform are saved beside the PNG.

The former zoomed artifacts remain under `outputs/subject_inspection/input`,
`inference`, and `meshes` solely as a comparison. **Use the `recommended_*`
directories below for the corrected exploratory export.**

## Measured mismatch

| Input versus its own refined DRR | NCC | SSIM | Edge NCC | Top-15% edge Dice |
| --- | ---: | ---: | ---: | ---: |
| Former zoomed `anchor_tighter` | 0.715 | 0.253 | 0.240 | 0.259 |
| Corrected masked `tighter_fov` | 0.846 | 0.316 | 0.448 | 0.379 |
| Published `04002` example | 0.873 | 0.522 | 0.699 | 0.560 |

The corrected crop materially improves global agreement and reproduces the
earlier sensitivity study's approximate `tighter_fov` values. It still has
substantially worse edge agreement than the published example, and the DRR
remains visibly smoother and different at the jaw. Flipping the corrected DRR
horizontally, vertically or 180° lowers NCC from `0.846` to `-0.187`, `0.581`
or `-0.278`, respectively; a simple display flip does not explain the gap.
These are image self-consistency metrics, **not** 3D accuracy scores.

The two subject crops produce meaningfully different 3D results: refined
volume correlation `0.964`, mean absolute intensity difference `0.0197`, and
warped-mandible Dice `0.809`. This alone invalidates treating a single crop
as a secure recovery of the patient's jaw. The corrected registered mandible
mask has 17,338 voxels, versus 18,251 in the former zoomed run.

The curated Go, constructed Go, Me, Pog and B marks have been mapped into the
corrected 128-pixel input and overlaid against a 2D projection of SC-DREG's
warped reference mandible mask at
`outputs/subject_drr_audit/mandible_mark_overlay.png`. Nearest mask-boundary
distances are recorded in `mandible_alignment.json`, but **a nearby outline
pixel is not the model's Gonion, a gonial angle, or FMA**. The available model
output does not supply reviewed corresponding 3D/2D landmarks. The user's
observed jaw-angle and vertical discrepancies are not contradicted by those
nearest-edge distances. We have not measured a model-minus-subject FMA angle;
reporting one from these masks would invent an unsupported landmark definition.

## Corrected files for exploratory viewing

All paths below are local and Git-ignored. Import PLY in Blender or 3D Slicer
without rescaling the skull and mandible relative to each other.

| Purpose | File |
| --- | --- |
| Corrected refined volume | `outputs/subject_inspection/recommended_inference/manual_masked_tighter_fov_recreated_128_refine.nii.gz` |
| Corrected intensity surface, 0.55 | `outputs/subject_inspection/recommended_meshes/subject_intensity_0p55.ply` |
| Corrected warped reference mandible | `outputs/subject_inspection/recommended_meshes/subject_warped_reference_mandible.ply` |
| Three grid-view previews | `outputs/subject_inspection/recommended_meshes/subject_intensity_0p55_views.png` |
| Input/DRR comparison | `outputs/subject_drr_audit/input_drr_comparison.png` |
| Mandible mark overlay | `outputs/subject_drr_audit/mandible_mark_overlay.png` |
| Numeric extraction and audits | `outputs/subject_inspection/recommended_meshes/extraction.json`, `outputs/subject_drr_audit/alignment.json`, `outputs/subject_drr_audit/mandible_alignment.json` |

The 0.55 PLY is only a visual intensity isosurface. The refined volume has
unit-range intensities, mean `0.16151` approximately; these are not HU.
The corrected sweep at 0.45/0.55/0.65 contains 186,528/110,197/77,232
voxels at or above threshold, respectively. Neither threshold identifies
validated bone. The separate mandible mesh is the **model's warped supplied
reference segmentation**, not a segmentation of this patient's CBCT.
NIfTI arrays use `[z,y,x]`; PLY vertices use `(x,y,z)` model indices. Header
unit spacing does not establish millimeters or anatomical axes.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts/prepare_subject_inspection_input.py
.venv\Scripts\python.exe scripts/run_sc_dreg_demo.py --input outputs/subject_inspection/recommended_input/manual_masked_tighter_fov_recreated_128.png --output-dir outputs/subject_inspection/recommended_inference --skip-reference-comparison --capture-deformation
.venv\Scripts\python.exe scripts/export_subject_inspection_meshes.py
.venv\Scripts\python.exe scripts/audit_subject_drr_alignment.py
.venv\Scripts\python.exe scripts/audit_subject_mandible.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The fixed-template deformation, incomplete clinical input calibration and lack
of paired 3D geometry prevent us from concluding that either exported mesh
matches the patient. Further image-score tuning would not establish the jaw's
true shape. Independent corresponding output landmarks or paired CBCT are
needed to measure and resolve the anatomical discrepancy.
