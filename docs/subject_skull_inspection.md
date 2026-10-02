# Subject SC-DREG skull for visual inspection

Status: exploratory model output from the local cephalogram, 2026-10-02.
This is an inspectable deformation of SC-DREG's fixed reference, not verified
patient-specific bone geometry.

## Input used

The exact prior ignored `manual/anchor_tighter_128.png`, private transform
matrix and mask config are absent in this checkout. The selected input is
`outputs/subject_inspection/input/manual_masked_anchor_tighter_recreated_128.png`.
It uses the same private ceph, the curated manual S/N/PNS/U1_tip/Me marks as
mapped checks, the documented mask for this image's vertical ruler, identity
intensity mapping and the **rounded** geometry reported for the prior manual
`anchor_tighter` run: center `(1799.64,1099.68)` source pixels, FOV width
`1603.57` source pixels and rotation `1.0502°`. That prior geometry came from
fitting the curated marks to the experimental `tighter_fov` template, then
applying the 1.15 tighter scale factor. We cannot recover its exact floating
point transform from the rounded report, so this is a recreation, not a
byte-exact copy of the prior manual candidate. The input config and invertible
**coordinate** transform are saved beside the PNG; crop/downsampling is not
invertible.

The ruler mask covers 0.893% of the source raster, but lies outside this
particular crop. Consequently the generated 128×128 PNG is pixel-identical to
the earlier approximate unmasked input in this checkout (SHA-256
`5157aea7b7591dfff43cee3665a8a1e16544a9fe14e66c0ba16e381b8f344178`).
Both inference runs produced exactly equal refined arrays. The current run
nevertheless uses the documented mask and curated manual landmark provenance.
The earlier report's manual registered mandible voxel count was 18,251; the
current count is also 18,251. That count alone does not establish exact
equality with the missing earlier reconstruction.

## Open these files

All files below are local, generated and Git-ignored. In Blender, use
**File > Import > Stanford PLY**; in 3D Slicer, use **Add Data**. Import the
skull and mandible PLY together without rescaling either one.

| Purpose | Local file |
| --- | --- |
| Refined subject volume | `outputs/subject_inspection/inference/manual_masked_anchor_tighter_recreated_128_refine.nii.gz` |
| Visual skull starting point | `outputs/subject_inspection/meshes/subject_intensity_0p55.ply` |
| Warped reference mandible | `outputs/subject_inspection/meshes/subject_warped_reference_mandible.ply` |
| Three grid-view previews | `outputs/subject_inspection/meshes/subject_intensity_0p55_views.png` |
| Full extraction measurements | `outputs/subject_inspection/meshes/extraction.json` |

The separately exported mandible is the **rounded SC-DREG warp of its supplied
reference mandible segmentation**. It is model-transferred anatomy, not an
independent segmentation of a subject CBCT. The refined NIfTI and PLY meshes
share model-grid coordinates. Array order is `[z,y,x]`, PLY vertex order is
`(x,y,z)`. The NIfTI unit spacing is a serialization default; imported viewers
may display that as millimeters, but no physical millimeter calibration or
anatomical axis orientation has been established.

## Intensity sweep

The refined volume is unit range (observed maximum `1.00000012` from floating
point interpolation), mean `0.16177`, median `0`, 75th percentile `0.34361`,
95th percentile `0.56846`. These values are not HU. Marching cubes exports:

| Isovalue | Voxels at or above | Vertices | Triangles | Mesh and views |
| ---: | ---: | ---: | ---: | --- |
| 0.45 | 189,486 | 154,619 | 307,146 | `subject_intensity_0p45.ply`, `subject_intensity_0p45_views.png` |
| **0.55** | **112,405** | **118,695** | **237,080** | `subject_intensity_0p55.ply`, `subject_intensity_0p55_views.png` |
| 0.65 | 78,992 | 101,597 | 202,586 | `subject_intensity_0p65.ply`, `subject_intensity_0p65_views.png` |

Start visual inspection at **0.55**: it gives a middle-density surface and is
near the intensity split used internally by the published DRR renderer. This
is a practical display choice, not a validated bone threshold. Compare 0.45
and 0.65 before interpreting a missing or merged structure. The warped
mandible mesh uses the binary registered mask's 0.5 boundary; it contains
12,503 vertices and 25,018 triangles. The PNG previews render subsampled
surface points from three **grid** views, not anatomical left/right labels or
screenshots from Blender/Slicer. The PLY files contain the full meshes.

## Reproduce

Install `requirements-exploration.txt` and the documented model assets, then
run from the repository root:

```powershell
.venv\Scripts\python.exe scripts/prepare_subject_inspection_input.py
.venv\Scripts\python.exe scripts/run_sc_dreg_demo.py --input outputs/subject_inspection/input/manual_masked_anchor_tighter_recreated_128.png --output-dir outputs/subject_inspection/inference --skip-reference-comparison --capture-deformation
.venv\Scripts\python.exe scripts/export_subject_inspection_meshes.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The ceph, curated marks, model assets, NIfTI and meshes stay in ignored local
directories. No independent CBCT is available to validate the reconstruction,
threshold or physical scale.
