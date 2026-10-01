# AGENTS.md

## Project objective

Build a reproducible pipeline that estimates craniofacial 3D geometry from a
lateral cephalometric radiograph, validates the reconstruction against known
cephalometric geometry, transfers the result to a high-resolution anatomical
mesh, and eventually produces a modular orthognathic demonstration model.

## Current priority

Do not work ahead of the active milestone.

The current milestone is:

> Reproduce SC-DREG inference on the published example and understand its input,
> projection, deformation, and output assumptions.

Do not begin Blender osteotomies, fixation plates, or model retraining until the
reproduction milestone is complete.

## Engineering principles

- Inspect upstream behavior before modifying it.
- Prefer evidence from source code, runtime inspection, paper, and model assets.
- Do not invent undocumented preprocessing assumptions.
- Clearly distinguish verified facts, strong inference, and unknowns.
- Keep compatibility changes separate from algorithmic changes.
- Preserve numerical semantics when modernizing dependencies.
- Avoid unnecessary rewrites.
- Prefer small, testable scripts over opaque notebooks.
- Make workflows reproducible from a fresh clone.
- Do not commit private medical data, generated volumes, or model weights.

## Upstream isolation

`vendor/sc-dreg/` is the upstream reference implementation.

Do not casually modify it.

If compatibility patches are needed, prefer project-side wrappers or clearly
documented patches.

Any behavior change must be validated against the upstream example.

## Model assets

Large pretrained assets live in `models/` and are excluded from Git.

Expected assets:

- cbct_c2f_model_ckpt.tar
- coeff4.npy
- mean4.npy
- ref.nii.gz
- seg_mandible.nii.gz
- xray_seg_unet_ckpt.tar

`coeff4.npy` is very large. Inspect it using NumPy memory mapping before loading
it conventionally.

Inference uses only the first `pca_dim` PCA components. Loading only the required
slice is allowed if it is mathematically identical to upstream behavior.

## Safety of private data

Private cephalograms, DICOM, CBCT, NRRD and other medical imaging files belong
under `data/private/` or `data/raw/`.

Never add them to Git.

Do not upload private data to external services unless explicitly requested.

## Definition of done for the first milestone

A fresh clone must be able to:

1. initialize the documented environment;
2. access the required local model assets;
3. run the published `04002.png` example;
4. produce the expected DRR and NIfTI outputs;
5. compare the outputs against the upstream reference;
6. report model/asset tensor shapes and data types;
7. document all compatibility modifications;
8. document known and unknown preprocessing assumptions.

When this is complete, stop and summarize the next logical milestone rather than
automatically expanding project scope.
