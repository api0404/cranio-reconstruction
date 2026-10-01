# Exploratory cephalogram alignment

This is separate from the published SC-DREG `04002` reproduction. The
upstream inference code requires a prepared 128 x 128 grayscale image but
does not specify a clinical image-to-model registration. The alignment below
is a **provisional geometric preview**, not a verified clinical input contract.

The user's local image is 2808 x 2136 RGBA with identical grayscale RGB
channels and fully opaque alpha. It shows the face to the right, matching
the published example, but includes a ruler in the upper-right background.
The PNG's roughly 96 DPI tag is not known detector calibration. The script
does not alter the original or remove the ruler.

## Method

`scripts/align_ceph_landmarks.py` reads explicitly supplied corresponding
2D points, fits `target = scale * source @ rotation + translation` by weighted
least squares, prohibits reflection, and samples the source once onto the
128 x 128 target canvas with bicubic interpolation. It does not shear or
scale x and y differently. A check-only landmark does not influence the fit.
It saves the transform, all residuals, a preview, and a 50/50 overlay for
review. The output is not passed to SC-DREG by this command.

The local provisional point file is under `data/private/` and is ignored by
Git. It contains three **visually estimated cranial regions** for fitting and
one menton-region check point. They were placed by visual inspection of the
private image and enlarged `04002.png`; they have not been produced or
validated by an anatomical landmark detector. The check point is held out
because fitting the patient's jaw to another subject's jaw would hide a
potential anatomical difference.

```powershell
.venv\Scripts\python scripts/align_ceph_landmarks.py data/private/sc_dreg_alignment_anchors.json
```

On the current provisional points, scale is 0.06035680 target pixels per
source pixel and rotation is 1.0502 degrees in image coordinates. Residuals
for the three fit points are 0.21, 0.61 and 0.42 target pixels. These small
numbers measure consistency of the **selected coordinates**, not landmark
accuracy. The held-out menton-region point is 11.27 pixels from its target
point, reflecting both landmark uncertainty and anatomical differences.
The overlay still contains the source ruler and shows substantial jaw/teeth
differences. The preview should not be interpreted as a successful patient
reconstruction or a verified model-ready clinical image.

The target is the published `04002.png`, which depicts another individual.
Once the SC-DREG baseline is reproduced, a fixed projection of the model
reference may be a preferable coordinate target; that choice must be
evaluated rather than assumed. A calibrated, overlay-free image and reviewed
landmarks would also improve this experiment. The original image and fitted
transform must be kept so any estimated geometry can be mapped back to its
source pixels.
