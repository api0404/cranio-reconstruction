# SC-DREG input-domain sensitivity on the local clinical ceph

Status: exploratory sensitivity study; **no paired CBCT ground truth and no
reviewed cephalometric landmark set**. The study measures input compatibility,
input–DRR self-consistency and variation of model outputs. It does not measure
patient-specific 3D accuracy or establish a clinically correct crop.

Run from the repository root after placing the private source image and local
configuration under `data/private/`:

```powershell
.venv\Scripts\python.exe scripts/study_sc_dreg_input_sensitivity.py
```

The local `data/private/sc_dreg_sensitivity.json` is a copy of the earlier
ignored `sc_dreg_alignment_anchors.json` with this explicit addition:

```json
"artifact_masks": [{
  "name": "vertical_ruler",
  "polygon_original": [[2618, 30], [2710, 30], [2710, 605], [2618, 605]],
  "fill_value": 0,
  "non_anatomical_reason": "Ruler in dark acquisition background, outside craniofacial contour"
}]
```

The original PNG is left unchanged and SHA-256 identified in
`outputs/sc_dreg_sensitivity/inputs/transforms.json`. The explicit polygon
replaces **only the vertical ruler body** in a copied image, affecting 0.893%
of source pixels. The source preview was visually checked: the polygon is in
the dark acquisition background, outside the apparent craniofacial contour;
the code also rejects any mask that contains a supplied landmark. The
horizontal bracket, some bright border marks and possible tiny ruler edges
remain. This is a manually reviewed artifact mask, not an automatic anatomy
segmentation or a claim that every artifact was removed. The mask itself and
the masked source are saved under ignored `outputs/`.

The only available landmarks are the previous **rough visual region
estimates** (`frontonasal_contour`, `sella_region`, `basion_region`, and a
held-out `menton_region`). They have not been reviewed as precise S/N/Ba/Me
points. The published `04002` image is a different person, so source-to-
template residuals are not clinical reprojection error. Independent landmark
marks on each output DRR do not exist; true 2D landmark reprojection error is
therefore **not measurable** in this study. The script records all source
landmarks projected by each explicit 2D transform and sets this error to
`null` rather than substituting image-feature proximity.

## Methods and interpretation

All 12 inputs use the same checkpoint, PCA basis, preserved upstream forward
semantics and CPU PyTorch 2.14.1. The model is loaded once. Cases are the
three existing canonicalizations, eight one-parameter perturbations about
`tighter_fov`, and `tighter_unmasked` with the identical transform. Sweep
values are ±3% uniform scale (inverse FOV), ±1° in-plane rotation, and ±2
model pixels horizontal or vertical translation. Intensity mapping remains
identity. Inference plus metrics and artifact writing took about 28–29 s on
this machine. No output volume is used to adjust an input transform.

NCC and SSIM compare the candidate input with the model's refined DRR on
the full 128 × 128 frame. Edge NCC compares Sobel gradient magnitudes;
edge Dice compares each image's strongest 15% gradient locations. These are
**self-consistency** measures and depend on background and contrast. The
training objective already encourages DRR agreement, so a high value is
not independent evidence of correct anatomy. The same metrics for coarse
DRRs and full input/DRR distributions are in `study.json`.

PCA statistics describe the model's 60 predicted coefficients. Deformation
magnitudes are Euclidean displacement lengths in the **128-grid voxel index
units**, not millimeters. Volume comparisons are elementwise in the common
model output grid. The mandibular proxy is the centroid of the registered
mandible mask thresholded at 0.5; the 2D proxy is the centroid of the
upstream segmentation projection thresholded at 0.5. Neither is an
anatomical landmark. NIfTI unit spacing is a serialization default, not a
clinical scale.

## Existing candidates

| Candidate | Source FOV px; outside input | Input–refined DRR NCC / SSIM / edge NCC / edge Dice | PCA L2; abs p95 | Coarse / refined displacement mean (p95), grid voxels | Volume mean / std | vs tighter: volume MAE / NCC; mandible Dice; 3D / projected centroid shift |
| --- | --- | --- | --- | --- | --- | --- |
| `tighter_fov` | 1844.1; 0% | 0.8463 / 0.3155 / 0.4483 / 0.3808 | 5.045; 1.451 | 1.610 (5.322) / 1.621 (5.369) | 0.16151 / 0.22632 | reference |
| `landmark_fit` | 2120.7; 3.65% | 0.8209 / 0.3330 / 0.1632 / 0.3068 | 4.784; 1.126 | 1.750 (5.853) / 1.759 (5.871) | 0.16190 / 0.22500 | 0.0252 / 0.9428; 0.7154; 2.92 grid vox / 3.70 px |
| `wider_fov` | 2495.0; 22.63% | 0.7054 / 0.2783 / 0.0891 / 0.2384 | 4.182; 1.258 | 1.715 (5.278) / 1.714 (5.318) | 0.16223 / 0.22474 | 0.0342 / 0.9057; 0.5914; 3.77 grid vox / 3.82 px |

The full `study.json` gives PCA min/max/mean/std and p05/p50/p95, coarse
and refined deformation max, volume min/max and high-density fraction,
segmentation voxel counts and centroids, and all pairwise metrics. The
reference-anchor fit errors are 0.21, 0.61 and 0.42 pixels by construction;
the held-out menton-region discrepancy is 11.27 pixels. These are rough
cross-person point comparisons, not measured anatomical accuracy.
The four rough source points move by a mean **5.37 model pixels** between
`tighter_fov` and `landmark_fit`, and **10.73 pixels** between `tighter_fov`
and `wider_fov` under the known 2D transforms. Those are input coordinate
changes, not independent DRR landmark detections. `study.json` records each
projected point and the pairwise mean/maximum shifts.

The much wider candidate has prominent zero-padded borders, and both NCC
and edge overlap fall. The tighter candidate's higher NCC and edge scores
support it as a **first controlled input hypothesis**, while `landmark_fit`
has a slightly higher SSIM. No single metric chooses a physically correct
FOV, and fitting landmarks to another person can conceal shape differences.

## Local perturbation around `tighter_fov`

| Perturbation | Input–refined NCC / SSIM / edge Dice | Volume MAE / NCC vs tighter | Mandible Dice | 3D / projected centroid shift |
| --- | --- | --- | --- | --- |
| scale −3% | 0.8629 / 0.3218 / 0.3983 | 0.0057 / 0.9959 | 0.9397 | 0.62 grid vox / 0.65 px |
| scale +3% | 0.8248 / 0.3074 / 0.3531 | 0.0051 / 0.9967 | 0.9528 | 0.42 / 0.50 |
| rotation −1° | 0.8445 / 0.3084 / 0.3812 | 0.0039 / 0.9981 | 0.9709 | 0.24 / 0.24 |
| rotation +1° | 0.8457 / 0.3187 / 0.3861 | 0.0045 / 0.9974 | 0.9696 | 0.28 / 0.31 |
| x −2 px | 0.8759 / 0.3443 / 0.4190 | 0.0103 / 0.9872 | 0.9182 | 0.95 / 1.24 |
| x +2 px | 0.8039 / 0.2794 / 0.3397 | 0.0080 / 0.9923 | 0.9663 | 0.21 / 0.24 |
| y −2 px | 0.8473 / 0.3009 / 0.3576 | 0.0146 / 0.9797 | 0.9016 | 1.03 / 1.41 |
| y +2 px | 0.8397 / 0.3224 / 0.3804 | 0.0117 / 0.9858 | 0.9056 | 0.98 / 1.27 |

Small changes do not leave the output invariant. The two-pixel shifts have
the largest volume and segmentation effects in this local sweep. The
`x_minus2px` probe has the highest self-consistency scores, but the study
cannot distinguish a true geometry improvement from the network's learned
preference or background alignment. It should **not** silently replace the
canonical transform.

With the same tighter transform and no ruler mask, input–refined NCC is
0.8426 vs 0.8463 masked, edge Dice is 0.3682 vs 0.3808, and the output
volume changes by MAE 0.0002 with mandible Dice 0.9988 and projected
centroid shift 0.016 px. Thus the marked vertical ruler has a small measured
effect for this candidate. The remaining bracket needs separate review.

## Recommendation and uncertainty

Use the **masked `tighter_fov`** candidate for the first exploratory
reconstruction, retaining `landmark_fit` and ±2-pixel translations as
sensitivity comparators. This combines lower artificial padding, stronger
NCC/edge agreement than the other existing FOVs, and better coverage of the
visible face and chin in the preview. Its SSIM is lower than `landmark_fit`,
and its estimated landmarks are not clinically reviewed. The broad-FOV
mandible Dice of 0.59–0.72 and the translation sensitivity preclude any
claim that this normalization recovers the patient's actual 3D anatomy.

Generated files under `outputs/sc_dreg_sensitivity/` are ignored. Inspect
`input_drr_montage.png`, case-level `input_drr_overlay.png`, and the
`volume_difference_tighter_vs_*.png` visualizations. The RGB overlays use
red for input, green for refined DRR, and yellow for rough source landmarks;
red in volume-difference images indicates larger voxel differences. The
three base cases also save refined NIfTI volumes and mandible masks. The
quantitative record is `study.json`. Private source/configuration and all
generated medical outputs remain untracked.
