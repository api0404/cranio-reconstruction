# SC-DREG baseline audit

Status (2026-10-01): the project-side runner and diagnostics are ready, but a
new inference run and comparison await `models/coeff4.npy`, still downloading.
The numerical values below describe the **committed upstream reference**, not
a reproduced output. Upstream revision: `8fd958ff195b3a4ebdc2bed9cc5bdb2c5fe25eb1`.

Sources: [`test.py`](../vendor/sc-dreg/test.py),
[`model.py`](../vendor/sc-dreg/src/model.py),
[`geometry.py`](../vendor/sc-dreg/src/geometry.py),
[`drr.py`](../vendor/sc-dreg/src/drr.py),
[`utils.py`](../vendor/sc-dreg/src/utils.py), and the
[published repository](https://github.com/Jyk-122/SC-DREG).

## Reproduce

On Windows with Python 3.11 (tested here with 3.11.9):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-sc-dreg.txt
.venv\Scripts\python scripts/inspect_model_assets.py
.venv\Scripts\python scripts/check_sc_dreg_checkpoint.py
.venv\Scripts\python scripts/check_sc_dreg_reference_drr.py
.venv\Scripts\python -m unittest discover -s tests
.venv\Scripts\python scripts/run_sc_dreg_demo.py
```

Place the six assets listed in the root README under `models/`. The runner
needs all except `xray_seg_unet_ckpt.tar`: the main checkpoint already contains
UNet weights, and upstream `test.py` does not call `load_pretrained()`. A
missing asset causes a clear error before loading PyTorch. Results go to
`outputs/sc_dreg_demo/`; `comparison.json` and console output contain MAE,
RMSE, maximum absolute error and exact element fraction against all five
committed reference outputs. The standalone comparison command is:

```powershell
.venv\Scripts\python scripts/compare_sc_dreg_outputs.py outputs/sc_dreg_demo
```

The pinned package set installed successfully on this CPU-only host:
PyTorch 2.14.1+cpu, torchvision 0.29.1, NumPy 2.4.6, SimpleITK 2.5.6,
Pillow 12.3.0 and imageio 2.38.0. Upstream pins PyTorch 1.8.1+cu101 and
NumPy 1.19.4, which are not suitable for this Python version. Full inference
speed and numerical agreement on this CPU remain unverified.

## Verified input and data flow

* The published input `vendor/sc-dreg/tests/04002.png` is 128 × 128 grayscale
  uint8, range 0–254. `read_img()` converts to `L`, then
  `torchvision.transforms.ToTensor()` scales to float32 in [0, 1]. `test.py`
  reshapes unconditionally to `[1, 1, 128, 128]`; another pixel count fails,
  while a non-square image of the same pixel count would be silently reshaped.
  The runner requires 128 × 128 explicitly. Inference has no crop, resize,
  intensity calibration, landmark alignment or left/right flip.
* `Reg23D` repeats the image into three channels, uses a ResNet-34 feature
  pyramid, and predicts 60 PCA coefficients. The coarse displacement is
  `para @ COEFF[:60] + mean_`, reshaped to `[1, 128, 128, 128, 3]`.
  `coeff4.npy` is a **deformation basis**, not an intensity basis; `mean4.npy`
  is the mean displacement field. The predicted field samples the reference
  CBCT and mandible segmentation.
* Image pyramid features are projected into the deformed 3D grid. `RefineNet`
  predicts a new dense three-channel displacement field using the coarse field
  and those features. The final volume samples the same reference with this
  new field. The code does not add the new field to the coarse one.
* `ref.nii.gz` is the float32 128³ intensity template in [0, 1].
  `seg_mandible.nii.gz` is a float32 128³ mandible template in [0, 1]. The
  warped segmentation is rounded at output. DRRs project warped intensities.
* Upstream `test.py` does **not** set evaluation mode or disable gradients.
  The project runner preserves training-mode BatchNorm and uses
  `torch.inference_mode()` to avoid storing gradients.

## Tensor and geometry conventions

* SimpleITK exposes images as NumPy `[z, y, x]`; PyTorch volumes are
  `[batch, channel, depth, height, width]` = `[1, 1, 128, 128, 128]`.
  The displacement grid is `[batch, depth, height, width, (x, y, z)]`.
  `mesh` assigns `(width, height, depth)` to its last axis. Displacements
  are in **voxel-index units**; `(grid - 63.5) / 63.5` maps to [-1, 1].
  Volume `grid_sample` uses bilinear/trilinear interpolation, zero padding
  and `align_corners=True`.
* `index_to_world()` permutes and negates normalized grid coordinates and maps
  one axis to a coded depth range of 101/128 to 117/128. `perspective()` divides
  by that depth with an identity calibration matrix; the image feature sampler
  uses `align_corners=True`. These are normalized model coordinates, with no
  physical projection matrix or millimeter calibration in the public code.
* DRR generation builds 128 depth-dependent affine grids, with scale increasing
  from 101/128 to 117/128, samples and sums the warped volume, splits
  air/soft/bone at 0 and 0.55, combines channels with fixed weights, rotates
  180 degrees and normalizes to [0, 1]. Its `affine_grid` and `grid_sample`
  omit an explicit `align_corners`; modern PyTorch defaults to `False` for
  both. The runner leaves these calls untouched.
* Source reference volumes have unit spacing, zero origin and identity
  direction. `save_mha()` creates a **new** SimpleITK image and does not copy
  source metadata. The committed NIfTI files have 1 × 1 × 1 spacing, zero
  origin, identity direction, float32 voxels and 128³ shape. This default
  array encoding does not establish real millimeter spacing or anatomy axes.

## Inspected assets and reference

| Asset | Observed data |
| --- | --- |
| `coeff4.npy` | Missing at audit time; shape and dtype unverified. At least 60 rows and 6,291,456 columns required. |
| `mean4.npy` | `(6291456,)`, float64, 50,331,776 bytes. |
| `ref.nii.gz` | `(128,128,128)`, float32, range 0–1. |
| `seg_mandible.nii.gz` | `(128,128,128)`, float32, range 0–1. |
| `cbct_c2f_model_ckpt.tar` | Epoch 29; 461 state tensors, 397 float32 and 64 int64; PCA head `(60,512)`. |
| `xray_seg_unet_ckpt.tar` | Epoch 499; 196 state tensors, 168 float32 and 28 int64; not loaded by `test.py`. |

`check_sc_dreg_checkpoint.py` constructs only the three network modules, with
no PCA or volume loaded. On PyTorch 2.14.1+cpu, strict loading reports **all
keys matched** for both checkpoints; the main network has 32,032,299 trainable
parameters. The standalone UNet's 112 parameter tensors are exactly equal to
the UNet embedded in the main checkpoint. Its other 84 buffers differ:
28 running means, 28 running variances, and 28 batch counters. This is
consistent with additional BatchNorm updates in main-model training.

The mean displacement has range -3.84294 to 3.21847 voxel-index units, mean
-0.05694 and standard deviation 0.62121. The source reference has 837,369
nonzero voxels; its mandible mask has 23,052 nonzero voxels and exactly two
values. The committed refined volume has 854,951 nonzero voxels, mean 0.15859,
and mean absolute voxel difference 0.037685 from the source reference.
The committed rounded mandible output has 16,565 nonzero voxels.

The committed `04002_raw.png` is pixel-identical to `04002.png`. The two DRRs
are 128 × 128 uint8 PNGs. The refined CBCT and rounded segmentation are 128³
float32 NIfTI images. Comparing the reference directory to itself yields
MAE = RMSE = max absolute error = 0 for every file. This validates the
comparison path, not new inference agreement.

The exact tensor from upstream `read_img('04002.png')` is bitwise equal to
the project runner's tensor: float32 `[128,128]`, min 0, max 254/255 and
mean 0.36354002. Re-encoding it with the project PNG writer gives pixels
identical to the committed raw output. The PNG has no embedded spatial or
acquisition metadata.

Without PCA, `check_sc_dreg_reference_drr.py` runs upstream `GenerateDRR`
on the committed refined NIfTI. On this CPU build, the regenerated DRR
differs from the committed PNG in **13/16,384 pixels**, MAE 0.0043335 gray
levels and maximum absolute difference 6. This isolates a small difference
in the projection path on the current CPU/PyTorch build, independent of
registration. Using `align_corners=True` for its volume sampler is much
worse: 10,584 differing pixels, MAE 1.61847 and maximum difference 29.
The runner retains the upstream default (`False`) for this sampler. The
source does not reveal which exact PyTorch/CUDA build produced the committed
PNG; the cause of the remaining 13-pixel difference is not established.

## Compatibility edits in project code

`scripts/sc_dreg_compat.py` reads upstream `model.py` and makes exact,
count-checked replacements **in memory**. Upstream files remain unchanged:

1. Construct ResNet-34 without downloading ImageNet weights; strict loading
   of the full saved state immediately follows.
2. Use the available device instead of unconditional `.cuda()`.
3. Memory-map `coeff4.npy`, cast only the first `pca_dim` rows to float32,
   and memory-map/cast the mean. Upstream `torch.Tensor(float64_array)` also
   casts to float32; rows beyond `pca_dim` are never read in `forward()`.
4. Load the checkpoint with explicit `weights_only=True` and `map_location`.
5. Encode output PNGs as uint8 because current Pillow/imageio reject float32
   PNGs. The conversion copies the float [0,1] rounding formula from
   [imageio 2.9's `image_as_uint`](https://github.com/imageio/imageio/blob/v2.9.0/imageio/core/util.py);
   image agreement still awaits a genuine run.

The wrapper checks source snippets so an upstream revision cannot silently
receive an incorrect patch. It retains training-mode BatchNorm, matching
`test.py`.

## Inference and unknowns

The arrays and network call strongly suggest `04002.png` is already normalized
into the model's expected field of view. Neither public inference code nor
the accompanying README establishes a clinical cephalogram conversion.
Crop boundaries, detector spacing, pose/orientation, intensity normalization
before PNG creation, anatomical calibration, and original CBCT voxel spacing
before downsampling are unknown. The 128 × 128 reshape alone is not a clinical
preprocessing recipe.

An October 2026 search of the [author's public repository](https://github.com/Jyk-122/SC-DREG)
and indexed records for the [journal paper](https://doi.org/10.1109/TMI.2024.3456251)
found no accessible author-provided clinical input preparation or supplementary
projection calibration procedure. The paper full text was not accessible in
that search, so this is an access limit rather than evidence that no such
procedure exists.

The next required step in **this milestone** is to place authentic `coeff4.npy`
in `models/`, run the demo, record its full shape/dtype, and diagnose any
generated DRR or volume discrepancy against the committed reference. The
baseline remains unreproduced until that check succeeds.
