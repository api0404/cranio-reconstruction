# Manual versus automatic landmark SC-DREG canonicalization

## Question and controlled setup

How much reconstruction variability arises when the **curated manual 2D
landmarks** are replaced by full-image py-ceph predictions? Both paths use
the same original 2808 × 2136 ceph, the same case-specific template, the
same source ruler mask, identity intensity mapping, Pillow bicubic sampling,
`anchor_tighter` scale factor 1.15, 60-component SC-DREG checkpoint and
model semantics. Only the five input fit marks change. The manual-landmark
run is the **experimental reference**, not anatomical 3D ground truth.

The comparison used the existing 128 × 128 candidate PNGs under ignored
`outputs/ceph_landmark_canonicalization/`, then ran each via
`scripts/run_sc_dreg_demo.py` with `--skip-reference-comparison`. The latter
is required because the published `04002` reference is a different subject.
The diagnostic script forwards both through the same compatibility wrapper
again and checks that refined NIfTI intensities and rounded segmentations
agree with the genuine demo runner. CPU, Python 3.11.9 and PyTorch 2.14.1+cpu
were used; individual forward times were 1.53 and 1.48 seconds. Upstream
SC-DREG remains unchanged and in training mode, matching the baseline.

## VERIFIED: transforms and 2D inputs

Pixel centers use x right/y down. Rotation is positive in that image
coordinate frame. Crop center is the inverse image location of model
`(63.5, 63.5)`. All distance units in this section are **original-image
pixels** or **128 × 128 model pixels**, not millimeters.

| Quantity | Manual | Automatic | Automatic minus manual |
| --- | ---: | ---: | ---: |
| Rotation | 1.0502° | 1.4989° | +0.4487° |
| Uniform scale, model px/source px | 0.079822 | 0.080834 | +1.268% |
| Square FOV width, source px | 1603.57 | 1583.49 | −20.08 (−1.252%) |
| Crop center x, source px | 1799.64 | 1804.89 | +5.24 |
| Crop center y, source px | 1099.68 | 1108.88 | +9.20 |

The crop-center displacement is **10.59 source pixels**. When exactly the
same curated source points are mapped under each transform, their model-space
shift ranges from 0.05 to 1.46 pixels across the curated set. This isolates
the global transform difference from detector point-label error.

| 128 × 128 input comparison | Value |
| --- | ---: |
| MAE on [0,1] scale | 0.04310 |
| RMSE on [0,1] scale | 0.06392 |
| Maximum absolute difference | 0.46667 |
| NCC | 0.95498 |

The input montage is ignored at
`outputs/manual_vs_pyceph/input_comparison.png`.

## VERIFIED: model-output differences

Both 128³ outputs are compared at identical model-grid indices. The
deformation field's three channels are `(x,y,z)` displacements in
voxel-index units; the volume is a unit-range reconstructed intensity, not
calibrated CT HU. The NIfTI unit spacing is a serialization default and
does not mean millimeters.

| Output | MAE | RMSE | NCC/correlation | Additional difference |
| --- | ---: | ---: | ---: | --- |
| PCA parameters, 60 values | 0.05650 | 0.07593 | 0.99181 | ΔL2 0.5882; manual L2 4.4440; auto L2 4.2607 |
| Coarse deformation components | 0.08229 | 0.14564 | 0.99231 | Vector Δ mean 0.1706, p95 0.5299, max 1.1428 grid units |
| Refined deformation components | 0.08396 | 0.15067 | 0.99189 | Vector Δ mean 0.1746, p95 0.5565, max 1.9619 grid units |
| Coarse DRR | 0.00814 | 0.01373 | 0.99841 | [0,1] intensity |
| Refined DRR | 0.00957 | 0.01510 | 0.99846 | [0,1] intensity |
| Refined volume | 0.007712 | 0.026566 | 0.993167 | Maximum absolute difference 0.6286 |

The largest PCA coefficient change is component 4: 2.3518→2.0399
(Δ−0.3119). The full 60 values are in ignored `comparison.json`.
The refined-volume differences are spatially uneven: 4.26% of voxels differ
by more than 0.05 and 1.71% by more than 0.1; only 36 voxels differ by more
than 0.5. The maximum is at array index `(z,y,x)=(25,103,74)` where
manual/automatic intensities are 0.9893/0.3607. The volume-difference
projection is under ignored
`outputs/manual_vs_pyceph/volume_difference_projection.png`.

The registered mandibular masks contain 18,251 and 18,829 voxels. Their
**Dice is 0.97087**. The centroid shifts by `(Δz,Δy,Δx)=(+0.0748,+0.2550,+0.0562)`
model-grid voxels, Euclidean displacement **0.2716 grid voxel**. There is
no physical millimeter interpretation.

## VERIFIED: detector errors and influence on the fit

Py-ceph errors below use the curated manual point as the 2D reference in
original-image pixels. The model-space column compares the manual point
mapped through the manual transform with the corresponding automatic point
mapped through the automatic transform; it combines mark and transform
changes.

| Point | Detector error, source px | Manual→auto mark displacement, model px | Used in fit? |
| --- | ---: | ---: | --- |
| S | 7.8 | 0.73 | Yes |
| N | 26.6 | 1.14 | Yes |
| PNS | 9.6 | 0.58 | Yes |
| U1_tip | 5.4 | 0.39 | Yes |
| Me | 8.3 | 0.35 | Yes |
| Po | 60.0 | 3.41 | No |
| ANS | 53.2 | 4.04 | No |
| Ar | 34.7 | 1.80 | No |
| Or | 29.8 | 2.19 | No |
| A | 26.7 | 1.89 | No |
| B | 2.4 | 0.11 | No |
| Pog | 16.1 | 1.49 | No |
| L1_tip | 23.6 | 1.82 | No |

Thus the largest skeletal/dental outliers, **Po and ANS, cannot directly
move this global transform** because they were not fit marks. They still
matter if used for subsequent landmark measurement or a future fit.
Within the five fit marks, N has the largest error. Replacing only the
automatic N with the manual N in a diagnostic fit moves the crop center by
5.39 source pixels and changes FOV by 15.82 source pixels. Replacing Me
changes rotation by 0.209°; S, PNS and U1_tip have smaller individual
effects. This substitution analysis is diagnostic only; the reported
automatic reconstruction did **not** use substituted marks or tuning to
match the manual reconstruction.

## VERIFIED: experimental ALI landmark movement

The existing optional ALI-CBCT probe was run on **both** refined volumes
with the same nine agents and the same explicitly hypothetical intensity
mapping `−1000 + 2000 × SC-DREG intensity`. Native ALI preprocessing would
cast these unit-range volumes to all zeros. ALI results are detector
annotations on synthetic reconstructions, not 3D anatomical truth.
The table compares the two ALI predictions, projects each through the
SC-DREG DRR geometry, then maps each projection through its own inverse ceph
transform.

| Landmark | 3D shift, model indices | Lateral shift, model px | Lateral shift, original px |
| --- | ---: | ---: | ---: |
| S | 0.784 | 0.874 | 14.37 |
| N | 0.292 | 0.288 | 7.97 |
| Ba | 0.430 | 0.505 | 11.57 |
| ANS | 0.354 | 0.412 | 7.67 |
| PNS | 0.850 | 0.997 | 5.78 |
| A | 0.071 | 0.059 | 5.55 |
| B | 0.461 | 0.541 | 4.26 |
| Pog | 0.800 | 0.939 | 11.41 |
| Me | 0.206 | 0.046 | 6.39 |
| **Mean** | **0.472** | **0.518** | **8.33** |

The difference in original-image pixels includes the changed inverse crop
transform. Small ALI movement might also reflect a shared anatomical prior
or ALI domain behavior; it is **not proof that the two 3D reconstructions
are anatomically equivalent**. The ignored
`outputs/manual_vs_pyceph/ali_projection_original_overlay.png` shows manual
in cyan and automatic in yellow.

## Reproduce

After the optional 2D detector and landmark canonicalizations from
[`ceph_2d_landmarks.md`](ceph_2d_landmarks.md) have run, execute from the
repository root:

```powershell
.venv\Scripts\python.exe scripts/run_sc_dreg_demo.py --input outputs/ceph_landmark_canonicalization/manual/anchor_tighter_128.png --output-dir outputs/manual_vs_pyceph/manual --skip-reference-comparison
.venv\Scripts\python.exe scripts/run_sc_dreg_demo.py --input outputs/ceph_landmark_canonicalization/automatic_full_stretch/anchor_tighter_128.png --output-dir outputs/manual_vs_pyceph/automatic --skip-reference-comparison
.venv-ali\Scripts\python.exe scripts/ali_cbct_probe.py --volume outputs/manual_vs_pyceph/manual/anchor_tighter_128_refine.nii.gz --output-dir outputs/manual_vs_pyceph/ali_manual --run --labels S N Ba ANS PNS A B Pog Me
.venv-ali\Scripts\python.exe scripts/ali_cbct_probe.py --volume outputs/manual_vs_pyceph/automatic/anchor_tighter_128_refine.nii.gz --output-dir outputs/manual_vs_pyceph/ali_automatic --run --labels S N Ba ANS PNS A B Pog Me
.venv\Scripts\python.exe scripts/compare_manual_auto_sc_dreg.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The two ALI commands are optional for the main SC-DREG sensitivity
comparison; if their `probe.json` files are absent, the comparison records
`ali: null`. `outputs/manual_vs_pyceph/comparison.json` contains full
precision, all manual and automatic point locations, all 60 PCA parameters,
individual fit-point influence and ALI projections when available. All
private input, model assets and generated medical output remain ignored.

## INFERRED assessment

For **automatic initial canonicalization of this specific ceph in the
current experimental template**, py-ceph appears adequate: replacing the
five manual fit marks yields a modest transform shift, refined-volume
correlation 0.9932, mandible Dice 0.9709 and subpixel mean ALI lateral
movement in model space. The displacement fields and localized volume
intensities do change, so manual marks remain preferable for a controlled
reference and for cephalometric measurements. This result does not justify
searching for another detector before the next controlled experiment, but
it also does not validate patient-specific 3D anatomy.

## UNKNOWN

The SC-DREG training crop/FOV, physical scale and true 3D geometry remain
unknown. The case-specific manual template came from a prior rough
`tighter_fov` normalization, not an author-supplied calibration. There is no
paired CBCT or independent ALI accuracy measurement for this subject. The
stability measured here may not hold for another ceph or a different set
of fit landmarks.
