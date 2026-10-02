# High-resolution 2D cephalometric landmark probe

## VERIFIED: detector and source

The original private ceph is 2808 × 2136 pixels. The ignored, curated
`data/private/high_value_ceph_landmarks.json` is the reference for this single
case. It contains source-pixel point coordinates, including S, N, ANS, PNS,
A, B, Pog, Me, U1 and L1 incisal tips. It supplies no physical pixel spacing
and no separate recorded angle table. The SNA/SNB/ANB values below are
**derived from those points**, not independent clinical measurements. The
manual JSON was read only after detector predictions were generated; its SHA
is recorded and it was not edited.

We screened [py-ceph](https://github.com/zhangted/py-ceph),
[CEPHMark-Net](https://github.com/manwaarkhd/CEPHMark-Net), and
[medical-landmark-detection](https://github.com/stolariks/medical-landmark-detection).
The local CEPHMark-Net checkout lacked an inference checkpoint. The local
medical-landmark-detection checkout lacked its documented evaluation weights.
The latter two were **not** benchmarked on this image. We integrated py-ceph
because its repository includes a pretrained model and runnable inference.
It is pinned to revision `9c2f7edfc8783c2621e61d1feada764bdc3c41fe`
in an ignored checkout, `vendor/py-ceph/`. The bundled checkpoint SHA-256 is
`ced18a57dab534e449433c84c1da6484044877eadb8c0c8157ed1b3b8b0ee55e`.
The project wrapper rejects a different checkpoint before invoking pickle.
The author's public `001.jpg` example returned exactly the README's 19 point
coordinates under this compatibility wrapper.

Py-ceph's `CephImage` converts to RGB, resizes to **800 rows × 640 columns**
with `skimage.transform.resize`, and returns integer `(x,y)` coordinates on
that grid. It reports 19 standard ISBI labels. The landmark mapping used here
is Sella→S, Nasion→N, Orbitale→Or, Porion→Po, Subspinale→A,
Supramentale→B, Pogonion→Pog, Menton→Me, Gonion→Go,
Incision Inferius/Superius→L1_tip/U1_tip, PNS, ANS, Articulare→Ar,
and four soft-tissue labels. The detector does **not** predict Ba, Co,
Go_constructed, incisor apices, molar cusps/apices, N_soft, Prn or Me_soft.
Go is excluded from the primary score because its clinical definition differs
across the available exports. Soft-tissue marks are also reported separately.

The worker uses CPU and leaves py-ceph source unchanged. Its old gzip pickle
serializes a `fusionVGG19` Python object under module `models`; the current
package exposes `pyceph.models`. We register that module name and explicitly
pass `weights_only=False` only for the pinned checkpoint. This is needed
because [PyTorch now defaults to restricted loading](https://docs.pytorch.org/docs/stable/notes/serialization).
The measured environment was PyTorch 2.14.1, NumPy 2.4.6,
scikit-image 0.26.0, matplotlib 3.11.2, Pillow 12.3.0; three predictions plus
one model load took 11.76 s on CPU.

## VERIFIED: comparison with the curated points

Coordinates were mapped from the detector's 640 × 800 grid back into original
2808 × 2136 source pixels before errors were measured. These are image-pixel
distances, **not millimeters**. The 13 comparable skeletal/dental marks form
the primary score. Each input strategy was chosen without reading the manual
points:

| Detector-only input | Mean px | Median px | Max px | RMSE px | SNA / SNB / ANB from predictions |
| --- | ---: | ---: | ---: | ---: | --- |
| Full image, author's resize | 23.42 | 23.64 | 60.04 | 29.13 | 80.15° / 78.27° / 1.88° |
| Full image, uniform letterbox | 127.63 | 38.71 | 1137.22 | 318.67 | 97.83° / 81.80° / 16.03° |
| Right-aligned portrait crop | 28.51 | 16.46 | 111.15 | 40.81 | 79.87° / 78.51° / 1.36° |
| Curated manual points | — | — | — | — | 78.78° / 76.56° / 2.22° |

The full-raster resize follows py-ceph's implementation and is **nonuniform**
for this landscape image. It is used for detection only. The SC-DREG
canonicalization below uses uniform scale, rotation and translation. The
full-raster SNA/SNB/ANB differences against angles derived from the manual
points are +1.37°/+1.71°/−0.34°; this does not erase point-level errors. The
letterbox caused a catastrophic A outlier; the portrait crop improved the
median but worsened Po (111.15 px) and ANS (71.3 px). Thus `full_stretch`
is the preferred **automatic initial annotation** for this case, subject to
manual review. Its largest primary errors are Po 60.04 px, ANS 53.2 px and
Ar 34.7 px; S 7.8 px, Me 8.3 px, PNS 9.6 px and U1_tip 5.4 px are closer.

| Point | Full resize | Letterbox | Portrait crop |
| --- | ---: | ---: | ---: |
| S | 7.8 | 32.8 | 29.4 |
| N | 26.6 | 56.2 | 16.5 |
| Or | 29.8 | 55.8 | 30.8 |
| Po | 60.0 | 38.7 | 111.1 |
| A | 26.7 | 1137.2 | 16.5 |
| B | 2.4 | 21.2 | 15.1 |
| Pog | 16.1 | 80.4 | 31.0 |
| Me | 8.3 | 35.6 | 7.8 |
| L1_tip | 23.6 | 10.6 | 11.4 |
| U1_tip | 5.4 | 37.2 | 5.5 |
| PNS | 9.6 | 69.8 | 3.4 |
| ANS | 53.2 | 46.0 | 71.3 |
| Ar | 34.7 | 37.7 | 21.1 |

`outputs/ceph_2d_detector/evaluation.json` holds full precision, all
supported soft-tissue/Go errors, hashes, coordinates and environment details.
`overlay_*.png` shows cyan manual crosses, yellow predictions and connecting
error lines on the untouched original. `automatic_*.json` is distinct from
the manual file and explicitly marked unreviewed.

## VERIFIED: landmark-driven 128 × 128 candidate generation

`scripts/canonicalize_from_ceph_landmarks.py` consumes either the curated
manual JSON or an automatic JSON, checks image hashes where available, and
builds a config for the existing reversible-coordinate canonicalizer. Its
**case-specific experimental template** takes the curated S, N, PNS,
U1_tip and Me coordinates through the prior sensitivity study's `tighter_fov`
matrix. That matrix was based on rough visual anchors, not author-specified
SC-DREG training geometry. Fitting the manual points recovers that matrix
within numerical precision. Fitting full-resize automatic points gives
1,821.0 source px FOV versus 1,844.1 px for manual; all five automatic
template residuals are below 1 model pixel. This small residual is a property
of fitting a similarity to a case-specific template, **not an independent
accuracy score**. The manual and automatic `anchor_fit` raster candidates
differ by 10.08/255 mean absolute intensity. The auto transform differs
because the marks differ; neither output replaces the original or the manual
points.

Each run saves 128 × 128 candidates, visual previews, an explicit generated
config, 3 × 3 forward/inverse matrices, round-trip landmark diagnostics,
source/template hashes, and a landmark provenance block. `anchor_wider` and
`anchor_tighter` perturb the fitted uniform scale by 0.85 and 1.15. The
vertical ruler mask is copied from the **private case-specific** sensitivity
config; it is not a generic mask for other patients. Only coordinate mapping
is invertible; raster crop/resampling and clipped intensities are not.

## Reproduce

In PowerShell, from the repository root, after the baseline `.venv` exists:

```powershell
git clone https://github.com/zhangted/py-ceph.git vendor/py-ceph
git -C vendor/py-ceph checkout 9c2f7edfc8783c2621e61d1feada764bdc3c41fe
uv venv --python .venv/Scripts/python.exe .venv-2d
Set-Content .venv-2d/Lib/site-packages/sc_dreg_baseline.pth (Resolve-Path .venv/Lib/site-packages).Path
uv pip install --python .venv-2d/Scripts/python.exe -r requirements-ceph-2d.txt
.venv/Scripts/python.exe scripts/run_ceph_2d_detector.py
.venv/Scripts/python.exe scripts/canonicalize_from_ceph_landmarks.py
.venv/Scripts/python.exe scripts/canonicalize_from_ceph_landmarks.py --landmarks outputs/ceph_2d_detector/automatic_full_stretch.json --output-dir outputs/ceph_landmark_canonicalization/automatic_full_stretch
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The last two canonicalization commands require the ignored original,
manual JSON, prior `outputs/sc_dreg_sensitivity/inputs/transforms.json`, and
case-specific mask config. The first detector command requires the original,
manual JSON, ignored py-ceph checkout and optional environment. No original
or private mark is committed or uploaded.

## INFERRED

The full-raster result suggests this detector can provide useful initial
marks for this ceph, but the outliers require human review. A different image
layout, scanner or orientation could change the ranking. Angles derived from
correlated predicted marks can appear close even when individual points are
displaced. The case-specific template is useful for a reproducible first
SC-DREG input; it does not establish the training normalization.

## UNKNOWN

The source image's pixel spacing and projection geometry remain unknown.
The manual point export does not include independently recorded measurement
values in this checkout. No detector uncertainty calibration or multi-case
accuracy estimate is available. No paired CBCT exists for this case; this
work does not measure patient-specific 3D accuracy.
