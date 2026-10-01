# Clinical ceph canonicalization for SC-DREG input experiments

This project-side pipeline creates **candidate inputs**, not a verified clinical
SC-DREG preprocessing standard. It does not claim patient-specific 3D accuracy.
No private image or generated candidate is committed or uploaded.

## VERIFIED evidence

- The author's [SC-DREG repository](https://github.com/Jyk-122/SC-DREG)
  supplies already prepared example images but no script for preparing a
  clinical ceph. In [`read_img`](../vendor/sc-dreg/src/utils.py), the image is
  converted to `L` and `ToTensor()` applies `uint8 / 255`; there is no crop,
  orientation correction, resize, histogram matching or physical calibration.
  [`test.py`](../vendor/sc-dreg/test.py) reshapes to `[1,1,128,128]` and
  [`datasets.py`](../vendor/sc-dreg/src/datasets.py) applies the same loader to
  training files. The model and DRR code use a fixed 128-pixel grid.
- The three commits in the vendored upstream history contain no separate
  preprocessing program. `tests/04002.png` entered with commit `9be7309`.
  It is 128 × 128, mode `L`, 8,866 bytes, SHA-256
  `0beaf11acec2b7794b56947aaeeceadaa9b3b7479e847a5880a46f0c53498f0f`.
  Its PNG has no EXIF, DPI, pixel-spacing, acquisition geometry or source
  identifier metadata. Its pixel range is 0–254, mean 92.70270, standard
  deviation 69.90251. The committed `04002_raw.png` is pixel-identical.
- The [SC-DREG paper record](https://doi.org/10.1109/TMI.2024.3456251)
  establishes the task as registration of a lateral cephalogram to a CBCT.
  The public abstract and repository do not supply a reproducible
  clinical-to-128×128 image recipe. We could not access the full IEEE text or
  identify author-provided supplementary preprocessing material in the public
  sources checked; this is an **access limit**, not proof it does not exist.
- The group's earlier [2021 Medical Physics paper](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.15214)
  reports 24,000 paired **synthetic** lateral ceph/CBCT samples generated from
  120 clinical CBCTs and evaluation on clinical lateral cephs. Its accessible
  abstract does not specify the crop, detector calibration or intensity
  mapping used by SC-DREG. This earlier dataset must not be assumed to be
  identical to SC-DREG's dataset. Their [2018 temporal registration work](https://doi.org/10.1007/978-3-030-00919-9_43)
  mentions a DRR pyramid, but its accessible abstract does not define a
  clinical input transform for the current model.
- [`drr.py`](../vendor/sc-dreg/src/drr.py) constructs normalized 128³
  sampling grids with depth-scale values from `101/128` to `117/128`. These
  are internal model coordinates. No mapping from ceph detector millimeters
  or a clinical X-ray acquisition to this grid is supplied.
- The local ceph is 2808 × 2136 RGBA with identical RGB channels and opaque
  alpha. Its ~96 DPI PNG tag is **not** verified detector pixel spacing. It
  shows the face to the right and a ruler in the upper-right background.

## INFERRED working approach

A spatial similarity transform is a conservative *test hypothesis* for
alignment: explicit left/right or top/bottom correction, rotation, uniform
scale and translation, followed by one square crop and downsampling. The
pipeline forbids non-uniform scaling and shear. It does not elastically warp
one person's anatomy to match the published example. A landmark fit to
`04002.png` is only an initialization because that image depicts a different
person, and no ground-truth anatomical correspondence is supplied.

Use reviewed, clearly defined cranial landmarks such as **Sella (S), Nasion
(N), and Basion (Ba)** for a small initialization set when visible. Check the
identifications on both images; use a held-out landmark such as Menton (Me)
to assess disagreement rather than forcing the patient's jaw onto the
template. If Ba is ambiguous, use a different reviewed cranial-base point
or explicit FOV placement; do not relabel a vague region as a precise point.
The current local anchors (`frontonasal_contour`, `sella_region`,
`basion_region`) are **visual region estimates**, not verified S/N/Ba
annotations.

## UNKNOWN

- Whether `04002.png` is a real radiograph, synthetic DRR, or a processed
  hybrid; its original pixels, crop, patient/source identifier and spacing.
- The SC-DREG training set's exact 2D crop/FOV, projection geometry,
  orientation convention, augmentation, resize kernel and intensity mapping.
- Whether the local clinical image's ruler provides an accurate millimeter
  calibration after export, and whether its projection magnification matches
  the training images.
- Which candidate FOV and intensity mapping generalizes to clinical images.
  Similar appearance or a plausible SC-DREG output does not establish this.

## Reversible coordinate pipeline

[`canonicalize_ceph.py`](../scripts/canonicalize_ceph.py) reads an original
grayscale PNG without changing it. Color images with unequal RGB channels
or transparency are rejected to avoid silent conversion choices. It builds
an explicit 3 × 3 `original_to_model` matrix in pixel-center coordinates
(x right, y down) and saves its `model_to_original` inverse. Each candidate
uses one Pillow bicubic spatial resampling to a 128 × 128 PNG. The JSON also
records the source and optional target hashes, Pillow version, selected
FOV corners, fraction of model samples outside the source, intensity mapping,
landmark residuals and round-trip coordinates. A source preview outlines the
FOV; model and published-example previews sit beside it with landmark labels.

The **coordinate map** is exactly invertible. Cropping and 128-pixel
downsampling discard image information; the PNG raster cannot be reconstructed
from the candidate. The original file is retained and SHA-256 identified.
Intensity modes are `identity`, `invert` or an explicitly specified affine
gain/offset. If affine values clip outside 0–255, the clipped fraction is
reported; clipping and uint8 quantization are not intensity-invertible.
No automatic histogram equalization is applied.

Candidate configuration supports:

- `orientation.flip_horizontal` and `flip_vertical`, each `true` only when
  a reviewed orientation correction is necessary; rotation is part of the
  candidate transform. No flip is used for the current local image.
- `landmark_similarity`: at least three source/target points marked
  `role: "fit"`, optional positive weights, and an explicit `scale_factor`
  around the 128-pixel center. A point marked `role: "check"` is reported
  but not fitted. Absent `candidates`, the script emits `landmark_fit`,
  `wider_fov` (0.85× fit scale) and `tighter_fov` (1.15× fit scale).
- `explicit_fov`: `center_oriented` in source pixel coordinates **after**
  any selected flip, `fov_width_px` for the square source FOV, and
  `rotation_degrees` in image coordinates. This path needs no landmarks.

Run the current local experiment from the repository root:

```powershell
.venv\Scripts\python.exe scripts/canonicalize_ceph.py data/private/sc_dreg_alignment_anchors.json
```

Outputs are ignored under `outputs/ceph_canonicalization/`. Review each
`*_preview.png` beside its `*_128.png` and `transforms.json`. The source
configuration is private and ignored. To test a candidate through the
reproduced model without comparing it to another subject's `04002` output:

```powershell
.venv\Scripts\python.exe scripts/run_sc_dreg_demo.py --input outputs/ceph_canonicalization/tighter_fov_128.png --output-dir outputs/ceph_canonicalization/tighter_fov_inference --skip-reference-comparison
```

That input path was exercised successfully on CPU. The resulting 3D volume
is an exploratory model output with **no clinical accuracy validation**. Its
saved raw PNG is pixel-identical to `tighter_fov_128.png` (0/16,384 pixels
differ), confirming that the inference loader accepted the candidate without
another geometric or intensity operation.
To map subsequently marked model-space 2D points back to original pixels,
write e.g. `[ {"name": "N", "model": [80, 36]} ]` in a local JSON and run:

```powershell
.venv\Scripts\python.exe scripts/map_ceph_landmarks.py outputs/ceph_canonicalization/transforms.json tighter_fov data/private/model_points.json
```

## Current candidate review and first test

The local rough-anchor fit yields a 2120.7-pixel square source FOV, with
3.65% of model pixel centers outside the original image. The wider FOV is
2495.0 source pixels and has 22.63% outside; it requires substantial padding.
The tighter FOV is 1844.1 source pixels and has **0% outside**. It retains the
face and chin in the preview but crops more of the posterior vault. It also
still includes part of the source ruler. The held-out `menton_region` differs
from its estimated template point by 10.67 pixels in this candidate, which
reflects both marking uncertainty and genuine patient differences.

**Test `tighter_fov` first as an input-compatibility and sensitivity probe**:
it avoids synthetic padding and follows the rough cranial-base alignment
without non-uniform warping. Compare it with `landmark_fit` only after
reviewing whether the latter's small padded region matters. Review true
landmark definitions and the ruler before drawing any anatomical conclusion.
Neither candidate can be selected as the clinically correct normalization
until training FOV/projection evidence or paired clinical ground truth exists.
