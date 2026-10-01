from pathlib import Path

import numpy as np


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


def main() -> None:
    inspect_npy("coeff4.npy")
    inspect_npy("mean4.npy")


if __name__ == "__main__":
    main()
