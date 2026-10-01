"""Quantitatively compare a SC-DREG run to the committed upstream example."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "vendor" / "sc-dreg" / "tests" / "output"
NAMES = (
    "04002_raw.png", "04002_coarse_drr.png", "04002_refine_drr.png",
    "04002_refine.nii.gz", "04002_seg_reg.nii.gz",
)


def compare(output_dir: Path, reference_dir: Path = REFERENCE) -> dict:
    results = {}
    for name in NAMES:
        generated = output_dir / name
        reference = reference_dir / name
        if not generated.is_file() or not reference.is_file():
            raise FileNotFoundError(f"Missing comparison file: {generated if not generated.is_file() else reference}")
        if name.endswith(".png"):
            a = np.asarray(Image.open(generated))
            b = np.asarray(Image.open(reference))
        else:
            a_img, b_img = sitk.ReadImage(str(generated)), sitk.ReadImage(str(reference))
            a, b = sitk.GetArrayFromImage(a_img), sitk.GetArrayFromImage(b_img)
            if (a_img.GetSpacing(), a_img.GetOrigin(), a_img.GetDirection()) != (b_img.GetSpacing(), b_img.GetOrigin(), b_img.GetDirection()):
                raise ValueError(f"NIfTI geometry differs: {name}")
        if a.shape != b.shape:
            raise ValueError(f"Shape differs for {name}: {a.shape} vs {b.shape}")
        diff = a.astype(np.float64) - b.astype(np.float64)
        results[name] = {
            "shape": list(a.shape), "dtype_generated": str(a.dtype), "dtype_reference": str(b.dtype),
            "mae": float(np.abs(diff).mean()), "rmse": float(np.sqrt(np.mean(diff * diff))),
            "max_abs": float(np.abs(diff).max()), "exact_fraction": float(np.mean(diff == 0)),
        }
        print(f"{name}: MAE={results[name]['mae']:.8g}, RMSE={results[name]['rmse']:.8g}, "
              f"max={results[name]['max_abs']:.8g}, exact={results[name]['exact_fraction']:.4%}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", default=ROOT / "outputs" / "sc_dreg_demo")
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE)
    args = parser.parse_args()
    compare(args.output_dir, args.reference_dir)
