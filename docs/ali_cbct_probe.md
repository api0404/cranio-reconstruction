# ALI-CBCT on SC-DREG volumes: compatibility experiment

This experiment uses [ALI-CBCT's published open-source detector](https://github.com/Maxlo24/ALI_CBCT) through the current [SlicerAutomatedDentalTools CLI](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/tree/main/ALI_CBCT). Its [published validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10440369/) was on native CBCT, with a reported 1.54 ± 0.87 mm mean 3D landmark error over the 32 validated landmarks. That result **does not transfer** to an SC-DREG reconstruction. No physical millimeter scale is established here.

## VERIFIED

- The inspected SlicerAutomatedDentalTools checkout is revision `e9411e92b2bbb5416f21cb8274442cc4353a74c3`; the earlier author repository checkout is `74da95ac5364f388ac0353c061027c57a2b32137`. Both are optional local source checkouts under ignored `vendor/` paths. SC-DREG upstream is unchanged.
- ALI's CLI clips image intensity near the 1st/99th histogram percentiles, caps the result at `[-1500, 4000]`, then casts to `int16`. Both SC-DREG refined volumes have roughly `[0,1]` float32 intensities. For 04002 the clipping interval is `[0.007, 0.998]`; **all 2,097,152 voxels are zero** after the cast, confirmed by running the upstream `CorrectHisto` itself. The subject volume also becomes all zero. Native ALI inference therefore has no usable image signal.
- ALI uses nominal `1` and `0.3` isotropic spacing, reorients the image to LPS, supplies `[z,y,x]` crops to separate DenseNet agents, and writes Slicer Markups LPS XYZ coordinates. The standard crop is `64³` with per-crop intensity scaling to `[-1,1]`. The input SC-DREG NIfTI is `128³`, identity direction, zero origin and unit spacing from serialization. Those units are **not calibrated millimeters**. ALI's fine resample becomes `426³` with origin approximately `(0.1,0.1,0.1)` in these arbitrary units.
- ALI's `SavePredictedLandmarks` uses `index*spacing - abs(origin)` instead of the physical `index*spacing + origin`. At the observed positive fine-grid origin, its exported position is offset by `-0.2` along each axis. `scripts/ali_cbct_probe.py` preserves the raw Markups and applies this measured coordinate correction in the project report; no upstream source is modified.
- The published cranial-base checkpoint archive has 10 landmark pairs of `1` and `0-3` models. Nine selected midline models across cranial base, upper bone and lower bone packages load and execute with Python 3.11, PyTorch 2.14.1+cpu, MONAI 1.3.2 and ITK 5.4.5 in an isolated optional environment. No checkpoint or PyTorch source patch was required.
- The explicit **proxy intensity hypothesis** `-1000 + 2000 * clip(SC-DREG,0,1)` gives ALI nonzero integer signal (2,096,491 voxels for 04002). It is *not* HU calibration. Under this hypothesis, S, N, Ba, ANS, PNS, A, B, Pog and Me agents completed on 04002 and the local subject. The 04002 markers are near visually plausible regions in the lateral input, but there are no independent 04002 landmark labels to quantify accuracy.
- The subject's curated source-image landmark export includes S, N, Ba, A, ANS, PNS, B, Pog and Me. It is independent of the rough visual anchors used for the `tighter_fov` candidate. The existing `original_to_model` matrix maps those coordinates into the `128×128` model plane; its inverse maps predictions back to the original ceph.
- SC-DREG's DRR sums along model X, using a depth-dependent scale from `101/128` to `117/128` for the Y/Z sampling grid, then rotates the resulting image 180°. `project_drr_xyz` in the probe reproduces this geometry; a synthetic single-voxel test against the upstream grid agreed within one output pixel. The nine subject predictions have X indices `63.15–65.55`, close to the model midplane, but this does not validate anatomical orientation.

## Landmark correspondence

| Curated 2D point | ALI model | Interpretation |
| --- | --- | --- |
| S, N, Ba | S, N, Ba in `Cranial_Base.zip` | Direct named midline correspondence; tested. |
| ANS, PNS, A | ANS, PNS, A in `Upper_Bones_v2.zip` | Direct named midline correspondence; tested. |
| B, Pog, Me | B, Pog, Me in `Lower_Bones_2.zip` | Direct named midline correspondence; tested. |
| Po, Or, Co, Go | R/L Po, Or, Co, Go | ALI has separate bilateral points; a single lateral 2D mark is not automatically one side's 3D point. |
| Ar | No matching ALI model in inspected label list | Unsupported. |
| U1/L1/U6/L6 tip, apex or cusp | Side-specific tooth occlusal/root labels | Definitions and side correspondence are not established; no direct scoring yet. |

The [extension label list](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/blob/main/ALI_CBCT/ALI_CBCT_utils/constants.py) and [checkpoint release](https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools/releases/tag/v0.1-v2.0_models) establish availability, not compatibility with synthetic volumes.

## Nine-landmark probe results

The project-side DRR projection uses SC-DREG's depth-dependent `affine_grid` scale and final 180° image rotation. It gives **model pixels**, not a calibrated radiographic projection. The subject input is the existing `tighter_fov` candidate. Yellow circles in the ignored overlay are ALI predictions and cyan crosses are curated 2D landmarks.

| Point | 04002 projected `(u,v)` | Subject projected `(u,v)` | Subject discrepancy (model px) | Subject discrepancy (original px) |
| --- | ---: | ---: | ---: | ---: |
| S | (48.6, 37.5) | (47.3, 34.0) | 3.80 | 54.7 |
| N | (96.7, 34.6) | (96.8, 33.8) | 10.86 | 156.4 |
| Ba | (24.4, 66.0) | (27.8, 63.3) | 5.08 | 73.2 |
| ANS | (96.6, 69.9) | (97.0, 68.1) | 14.51 | 209.0 |
| PNS | (60.8, 69.4) | (62.9, 69.0) | 3.88 | 55.9 |
| A | (93.7, 75.6) | (93.7, 73.9) | 8.77 | 126.4 |
| B | (88.0, 105.0) | (94.7, 106.3) | 6.98 | 100.6 |
| Pog | (94.5, 105.3) | (94.6, 106.6) | 6.78 | 97.7 |
| Me | (86.5, 116.2) | (91.1, 113.7) | 5.30 | 76.3 |

Mean subject discrepancy is `7.33` model pixels (median `6.78`, range `3.80–14.51`). The mean target-minus-prediction horizontal difference is `+5.72` pixels; subtracting that shared translation lowers 2D RMS from `8.05` to `5.66` pixels, so a global offset alone does not resolve the mismatch. Across 04002 and subject, corresponding detected 3D points shift `0.70–5.88` **model indices**; the reconstructed volumes themselves have MAE `0.0240` and correlation `0.9461`. This limited movement may reflect model prior or shared shape and should not be interpreted as anatomical stability.

The exact unrounded values, raw Markups, corrected model coordinates, SHA-256 provenance, transform results, and model/full-resolution overlays are under ignored `outputs/ali_probe/`. The nine-label ALI search took `148.77 s` on 04002 and `136.68 s` on the subject on CPU; this excludes preprocessing and process startup.

## Reproduce

Use the existing `.venv` for SC-DREG. The optional `.venv-ali` reuses its installed PyTorch/NumPy/SimpleITK via a local `.pth` file and holds ALI-specific packages separately. This local setup uses PyTorch `2.14.1+cpu`; no CUDA device is available. From the repository root:

```powershell
uv --cache-dir outputs/uv-cache venv --python .venv/Scripts/python.exe .venv-ali
Set-Content -LiteralPath .venv-ali/Lib/site-packages/sc_dreg_baseline.pth -Value (Resolve-Path .venv/Lib/site-packages).Path
uv --cache-dir outputs/uv-cache pip install --python .venv-ali/Scripts/python.exe --no-deps -r requirements-ali-cbct.txt
git clone https://github.com/DCBIA-OrthoLab/SlicerAutomatedDentalTools.git vendor/slicer-automated-dental-tools
git -C vendor/slicer-automated-dental-tools checkout e9411e92b2bbb5416f21cb8274442cc4353a74c3
gh release download v0.1-v2.0_models --repo DCBIA-OrthoLab/SlicerAutomatedDentalTools --pattern Cranial_Base.zip --pattern Upper_Bones_v2.zip --pattern Lower_Bones_2.zip --dir models/ali-cbct
```

The `.venv-ali`, ALI source checkout, archives and extracted weights are ignored. The probe extracts only requested model pairs from the archives and fails clearly if a required archive is missing. The three archives total about 3.24 GB; S/N/Ba need only `Cranial_Base.zip`.

```powershell
.venv\Scripts\python.exe scripts/ali_cbct_probe.py
.venv-ali\Scripts\python.exe scripts/ali_cbct_probe.py --run --labels S N Ba ANS PNS A B Pog Me
.venv\Scripts\python.exe scripts/analyze_ali_cbct_probe.py outputs/ali_probe/04002/probe.json vendor/sc-dreg/tests/04002.png

.venv-ali\Scripts\python.exe scripts/ali_cbct_probe.py --volume outputs/sc_dreg_sensitivity/cases/tighter_fov/refined.nii.gz --output-dir outputs/ali_probe/subject_tighter_fov --run --labels S N Ba ANS PNS A B Pog Me
.venv\Scripts\python.exe scripts/analyze_ali_cbct_probe.py outputs/ali_probe/subject_tighter_fov/probe.json outputs/sc_dreg_sensitivity/inputs/tighter_fov_128.png --transforms outputs/sc_dreg_sensitivity/inputs/transforms.json --curated-landmarks data/private/high_value_ceph_landmarks.json --original-image data/private/Cephalometric_X-Ray.png
```

The probe uses `--run` deliberately: the default action measures native/prepared intensity behavior without invoking ALI or requiring its models. `--inspect-existing` recalculates corrected coordinates from an existing ALI output without rerunning the agents. The project scripts leave private images and all medical/model outputs ignored.

## INFERRED

- The 04002 projected midline positions look regionally plausible on a 128-pixel lateral image. This is only a visual feasibility check.
- Some of the subject discrepancy may arise from SC-DREG reconstruction bias, FOV normalization, ALI domain shift, coordinate conventions or imperfect 3D-to-2D landmark equivalence. These contributions cannot be separated from the available data.
- The relatively small 04002-to-subject landmark shifts could reflect shared global anatomy or a strong ALI location prior; they are not evidence of individual accuracy.

## UNKNOWN

- SC-DREG's true physical spacing, orientation and projection calibration; hence ALI's nominal spacing has no physical meaning on these volumes.
- Whether the proxy contrast matches any part of ALI's training intensity distribution.
- 3D landmark accuracy on either reconstruction. The local subject has no paired CBCT; the upstream 04002 example has no independent landmark labels.
- Which bilateral ALI landmark, if either, corresponds to an overlapped 2D lateral cephalometric mark.

These predictions should be treated as experimental annotations for diagnosing the SC-DREG output. They are not patient-specific ground truth.
