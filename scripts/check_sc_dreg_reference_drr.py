"""Regenerate the committed refined DRR from its volume without the PCA basis."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch

from run_sc_dreg_demo import ROOT
from sc_dreg_compat import UPSTREAM_SRC

sys.path.insert(0, str(UPSTREAM_SRC))
from drr import GenerateDRR  # noqa: E402


def main() -> None:
    output = ROOT / "vendor/sc-dreg/tests/output"
    volume = sitk.GetArrayFromImage(sitk.ReadImage(str(output / "04002_refine.nii.gz")))
    tensor = torch.from_numpy(volume).reshape(1, 1, 128, 128, 128)
    with torch.inference_mode():
        drr = GenerateDRR()(tensor).squeeze().numpy()
    if not np.isfinite(drr).all() or drr.min() < 0 or drr.max() > 1:
        raise ValueError("Regenerated DRR is not finite [0,1]")
    generated = (drr.astype(np.float64) * 255 + 0.499999999).astype(np.uint8)
    reference = np.asarray(Image.open(output / "04002_refine_drr.png"))
    diff = generated.astype(np.int16) - reference.astype(np.int16)
    print(f"PyTorch {torch.__version__}; CUDA available={torch.cuda.is_available()}")
    print(f"DRR regenerated from committed refined volume: "
          f"MAE={np.abs(diff).mean():.8g}, max={np.abs(diff).max()}, "
          f"differing_pixels={np.count_nonzero(diff)}/{diff.size}")


if __name__ == "__main__":
    main()
