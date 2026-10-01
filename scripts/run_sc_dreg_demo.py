"""Run the published 04002 SC-DREG example using project-side compatibility edits."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from compare_sc_dreg_outputs import compare
from sc_dreg_compat import ROOT, load_model_class


ASSET_NAMES = ("cbct_c2f_model_ckpt.tar", "coeff4.npy", "mean4.npy", "ref.nii.gz", "seg_mandible.nii.gz")


def validate_inputs(models: Path, image: Path, pca_dim: int) -> None:
    if pca_dim != 60:
        raise ValueError(f"Published checkpoint has a 60-output PCA head; got --pca-dim={pca_dim}")
    missing = [str(models / name) for name in ASSET_NAMES if not (models / name).is_file()]
    if missing:
        raise FileNotFoundError("Required SC-DREG asset(s) missing:\n  " + "\n  ".join(missing))
    if not image.is_file():
        raise FileNotFoundError(f"Example input missing: {image}")
    with Image.open(image) as im:
        if im.size != (128, 128):
            raise ValueError(f"SC-DREG requires a 128x128 input; got {im.size}")
    try:
        coeff = np.load(models / "coeff4.npy", mmap_mode="r", allow_pickle=False)
        mean = np.load(models / "mean4.npy", mmap_mode="r", allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise ValueError(f"Cannot memory-map SC-DREG PCA assets (possibly incomplete download): {exc}") from exc
    if coeff.ndim != 2 or coeff.shape[0] < pca_dim or coeff.shape[1] != 128**3 * 3:
        raise ValueError(f"Unexpected coeff4.npy shape {coeff.shape}; need at least ({pca_dim}, {128**3 * 3})")
    if mean.shape != (128**3 * 3,):
        raise ValueError(f"Unexpected mean4.npy shape {mean.shape}")
    if coeff.dtype.kind != "f" or mean.dtype.kind != "f":
        raise ValueError(f"PCA arrays must be floating point; got coeff={coeff.dtype}, mean={mean.dtype}")
    print(f"coeff4.npy: shape={coeff.shape}, dtype={coeff.dtype}; loading first {pca_dim} rows")
    print(f"mean4.npy: shape={mean.shape}, dtype={mean.dtype}")


def save_png(path: Path, tensor) -> None:
    # imageio 2.9 image_as_uint: float [0,1] -> floor(float64 * 255 + 0.499999999).
    array = tensor.detach().squeeze().cpu().numpy()
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError(f"Expected finite [0,1] PNG data for {path.name}")
    pixels = (array.astype(np.float64) * 255 + 0.499999999).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--input", type=Path, default=ROOT / "vendor" / "sc-dreg" / "tests" / "04002.png")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "sc_dreg_demo")
    parser.add_argument("--pca-dim", type=int, default=60)
    args = parser.parse_args()
    try:
        validate_inputs(args.models, args.input, args.pca_dim)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    import torch
    import SimpleITK as sitk

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name() if device.type == "cuda" else "CPU (CUDA unavailable)"
    print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, "
          f"CUDA build={torch.version.cuda}, device={device}: {device_name}", flush=True)
    model = load_model_class()(args.pca_dim, str(args.models)).to(device)
    checkpoint = torch.load(args.models / "cbct_c2f_model_ckpt.tar", map_location=device, weights_only=True, mmap=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    # Upstream test.py never calls eval(); retain training-mode BatchNorm behavior.
    input_array = np.asarray(Image.open(args.input).convert("L"), dtype=np.float32) / 255.0
    img = torch.from_numpy(input_array).reshape(1, 1, 128, 128).to(device)
    with torch.inference_mode():
        outputs = model(img)
    coarse_drr, refine_drr, _, _, _, _, refine_volume, _, _, _, _, refine_vol_seg = outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    for suffix, tensor in (("raw", img), ("coarse_drr", coarse_drr), ("refine_drr", refine_drr)):
        save_png(args.output_dir / f"{stem}_{suffix}.png", tensor)
    for suffix, tensor in (("refine", refine_volume), ("seg_reg", torch.round(refine_vol_seg))):
        volume = tensor.detach().squeeze().cpu().numpy()
        sitk.WriteImage(sitk.GetImageFromArray(volume), str(args.output_dir / f"{stem}_{suffix}.nii.gz"))
    metrics = compare(args.output_dir)
    (args.output_dir / "comparison.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
