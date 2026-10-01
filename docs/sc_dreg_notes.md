# SC-DREG baseline audit

Status (2026-10-01): the published `04002` example has run end-to-end on this
machine and agrees very closely with the committed upstream outputs. The
remaining DRR pixel differences and their likely numerical cause are below.
Upstream revision: `8fd958ff195b3a4ebdc2bed9cc5bdb2c5fe25eb1`.

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
`outputs/sc_dreg_demo/`; `comparison.json` includes descriptive statistics,
MAE, RMSE, maximum absolute error, correlation and differing-element counts
against all five committed outputs. The standalone comparison command is:

```powershell
.venv\Scripts\python scripts/compare_sc_dreg_outputs.py outputs/sc_dreg_demo
```

The pinned package set installed successfully on this CPU-only host:
PyTorch 2.14.1+cpu, torchvision 0.29.1, NumPy 2.4.6, SimpleITK 2.5.6,
Pillow 12.3.0 and imageio 2.38.0. Upstream pins PyTorch 1.8.1+cu101 and
NumPy 1.19.4, which are not suitable for this Python version.

## VERIFIED: end-to-end run and numerical comparison

The authentic `coeff4.npy` opened via NumPy memory mapping and the project
runner completed `04002` with strict checkpoint loading. The machine has an
AMD Radeon RX 5700 XT and no NVIDIA CUDA device; the measured run used
Python 3.11.9, PyTorch 2.14.1+cpu, torchvision 0.29.1 and NumPy 2.4.6.
In the externally monitored run, forward inference took **1.70 s**, the
complete runner **9.09 s**, and process wall time **10.42 s**. A final direct
rerun took **1.84 s forward / 11.14 s total** and produced identical metrics.
The sampled peak process-tree working
set was **4.40 GiB** (3.42 GiB private bytes); Windows reported a 4.63 GiB
peak working set for the main Python process. The memory monitor followed the
venv launcher's child process, not just the small launcher stub.

All values below are for generated outputs compared elementwise with the
committed upstream files. `mean/std` describe the generated array; the JSON
report includes both generated and reference distributions.

| Output | Shape, dtype | Generated min/max; mean/std | MAE; RMSE; max abs | Correlation | Different elements |
| --- | --- | --- | --- | --- | --- |
| raw PNG | 128 x 128, uint8 | 0/254; 92.70270/69.90251 | 0; 0; 0 | 1 | 0/16,384 |
| coarse DRR PNG | 128 x 128, uint8 | 0/255; 92.16620/61.61890 | 0.00494385; 0.158384; 6 | 0.999996697 | 16/16,384 (0.0977%) |
| refined DRR PNG | 128 x 128, uint8 | 0/255; 91.78290/62.75318 | 0.0078125; 0.222348; 12 | 0.999993725 | 22/16,384 (0.1343%) |
| refined NIfTI | 128 x 128 x 128, float32 | 0/1.000000119; 0.158590902/0.224745124 | 5.08239e-8; 2.68605e-7; 1.52588e-5 | 0.999999999999258 | 422,820/2,097,152 (20.1616%) |
| segmentation NIfTI | 128 x 128 x 128, float32 | 0/1; 0.007898808/0.088523536 | 0; 0; 0 | 1 | 0/2,097,152 |

The many bitwise differences in the refined volume are tiny floating-point
differences: only 38 voxels differ by more than 1e-5. The segmentation is
exactly equal after upstream's rounding step. Both generated NIfTI files
have unit spacing, zero origin and identity direction, exactly as the
committed outputs; those defaults are not physical calibration.

The full PCA file is `(129, 6291456)` float64, 6,492,782,720 bytes on disk
(6.047 GiB logical array data). `pca_dim=60` maps the file and casts only
its first 60 rows to float32, yielding a 1.406 GiB materialized basis.
The unused 69 rows are not copied. The file's SHA-256 on this machine is
`621623b823b063e766e67466c9f86dee75f9e959a29ad5113c90abff96d086a8`.

## INFERRED: cause of the remaining differences

The generated refined PNG can be regenerated **exactly** from the generated
NIfTI using the current CPU DRR code and the imageio 2.9 quantization rule.
Regenerating from the committed upstream NIfTI with the same CPU DRR differs
from its committed PNG at 13 pixels (MAE 0.0043335, max 6). The two saved
volumes, projected by the same CPU code, differ at 9 pixels (MAE 0.0034790,
max 12). These two disjoint pixel sets account for the 22 differing refined
PNG pixels. Each of the 9 volume-driven pixels has one or two projection
samples that cross the hard `<= 1` bone threshold due to tiny floating-point
changes. The other 13 pixels are consistent with sampling/accumulation
rounding around that threshold between the original execution environment
and current CPU PyTorch, but the exact original CUDA/PyTorch build is unknown.
This is a likely numerical explanation, not proof of the original build.
The coarse DRR also has only sparse differences of 5–6 gray levels, but the
repository contains no committed coarse 3D volume to decompose those pixels
in the same way.

Changing the DRR volume sampler to `align_corners=True` produces 10,584
differing pixels (MAE 1.61847, max 29) even when projecting the same
committed volume. The upstream omitted argument must be preserved as its
effective `False` default. The near-exact 3D volume and exact segmentation
also argue against a basis-order, orientation, or checkpoint mismatch.

## UNKNOWN: source-data calibration

The physical projection geometry and clinical cephalogram crop, scale,
orientation and intensity preparation remain undocumented by the public
inference implementation. The upstream code does not encode a millimeter
voxel spacing or a detector calibration in its output NIfTI. The exact
environment used to generate the committed reference outputs is also
unknown.

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
| `coeff4.npy` | `(129,6291456)`, float64, 6,492,782,720 bytes on disk; first 60 rows used. |
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
MAE = RMSE = max absolute error = 0 for every file. This served as an initial
comparison-tool check; the real run results are reported above.

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
   the genuine raw PNG agrees exactly with the committed reference.

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

The published example is now reproduced closely enough to serve as the
project's measured SC-DREG baseline. The exploratory alignment in
`docs/experimental_ceph_alignment.md` remains separate and does not validate
any private patient's 3D reconstruction.
