# SC-DREG Notes

Upstream repository:

https://github.com/Jyk-122/SC-DREG

## Known pretrained assets

Expected under `models/`:

- `cbct_c2f_model_ckpt.tar`
- `coeff4.npy`
- `mean4.npy`
- `ref.nii.gz`
- `seg_mandible.nii.gz`
- `xray_seg_unet_ckpt.tar`

These files are intentionally excluded from Git.

## Initial source inspection

- Input image: 128 x 128 grayscale
- Reference volume: 128 x 128 x 128
- Default PCA dimensionality: 60
- SC-DREG predicts PCA parameters followed by a dense 3D deformation refinement
- The source expects input PNGs to already have the correct geometry
- No general resize/crop/alignment preprocessing is present in the public
  inference code
- The output NIfTI writer does not appear to preserve meaningful physical
  spacing/origin/direction metadata
- Projection geometry is hard-coded and requires careful documentation
- `grid_sample` / `align_corners` behavior must be preserved during runtime
  modernization

## First milestone

1. Reproduce upstream inference on `04002.png`
2. Verify checkpoint compatibility
3. Inspect all pretrained asset shapes and data types
4. Compare generated outputs against upstream example outputs
5. Document expected input geometry and unresolved assumptions
