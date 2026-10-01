from pathlib import Path

from collections import Counter

import numpy as np
import SimpleITK as sitk
import torch


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def inspect_npy(name: str) -> None:
    path = MODEL_DIR / name

    if not path.exists():
        print(f"{name}: missing")
        return

    array = np.load(path, mmap_mode="r")

    print(
        f"{name}: "
        f"shape={array.shape}, "
        f"dtype={array.dtype}, "
        f"size={path.stat().st_size / 1024**3:.3f} GiB"
    )


def inspect_volume(name: str) -> None:
    path = MODEL_DIR / name
    if not path.is_file():
        print(f"{name}: missing")
        return
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayViewFromImage(image)
    print(f"{name}: shape={array.shape}, dtype={array.dtype}, "
          f"range=({float(np.min(array)):.6g}, {float(np.max(array)):.6g}), "
          f"spacing={image.GetSpacing()}, origin={image.GetOrigin()}, "
          f"direction={image.GetDirection()}")


def inspect_checkpoint(name: str) -> None:
    path = MODEL_DIR / name
    if not path.is_file():
        print(f"{name}: missing")
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["model_state_dict"]
    dtypes = Counter(str(t.dtype) for t in state.values())
    print(f"{name}: epoch={checkpoint.get('epoch')}, tensors={len(state)}, "
          f"dtypes={dict(dtypes)}, parameter_bytes={sum(t.numel() * t.element_size() for t in state.values())}")
    for key in ("reg23d.fc1.weight", "reg23d.fc1.bias", "refinenet.conv1.0.weight", "unet.down_1.0.0.weight"):
        if key in state:
            print(f"  {key}: shape={tuple(state[key].shape)}, dtype={state[key].dtype}")


def main() -> None:
    inspect_npy("coeff4.npy")
    inspect_npy("mean4.npy")
    inspect_volume("ref.nii.gz")
    inspect_volume("seg_mandible.nii.gz")
    inspect_checkpoint("cbct_c2f_model_ckpt.tar")
    inspect_checkpoint("xray_seg_unet_ckpt.tar")


if __name__ == "__main__":
    main()
