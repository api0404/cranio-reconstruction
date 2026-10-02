"""Isolated worker for the published py-ceph checkpoint and inference code.

Called by run_ceph_2d_detector.py with the optional .venv-2d interpreter.
The pinned checkpoint serializes a Python model object, so loading it requires
weights_only=False. Only use the published checkpoint from the pinned checkout.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


PUBLISHED_CHECKPOINT_SHA256 = "ced18a57dab534e449433c84c1da6484044877eadb8c0c8157ed1b3b8b0ee55e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pyceph-root", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str((args.pyceph_root / "src").resolve()))
    import torch
    import pyceph.models as model_definitions

    # The author's pickle names its class `models.fusionVGG19`; the package
    # now places that class at `pyceph.models.fusionVGG19`.
    sys.modules["models"] = model_definitions
    from pyceph.CephImageBatch import CephImage

    checkpoint = args.pyceph_root / "src/pyceph/pretrained_models/12-26-22.pkl.gz"
    if not checkpoint.is_file():
        parser.error(f"Missing published py-ceph checkpoint: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != PUBLISHED_CHECKPOINT_SHA256:
        parser.error("Published py-ceph checkpoint SHA-256 mismatch; refusing pickle load")
    started = time.perf_counter()
    with gzip.open(checkpoint, "rb") as source:
        model = torch.load(source, map_location="cpu", weights_only=False)
    model.eval()
    configuration = SimpleNamespace(image_scale=[800, 640], R2=41, use_gpu="cpu")
    inputs = json.loads(args.manifest.read_text(encoding="utf-8"))
    predictions = {}
    for name, path in inputs.items():
        image = CephImage(str(path))
        if image.image.ndim != 3 or image.image.shape[2] != 3:
            raise ValueError(f"py-ceph requires 3-channel input: {path} {image.image.shape}")
        image.process(model, configuration)
        predictions[name] = {key: list(value) for key, value in image.to_dict().items()}
    result = {
        "detector": "py-ceph bundled fusionVGG19 / 19-landmark checkpoint",
        "network_shape_hw": [800, 640],
        "voting_radius": 41,
        "device": "cpu",
        "runtime_seconds_including_model_load": time.perf_counter() - started,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "numpy", "scikit-image", "matplotlib", "Pillow")},
        "predictions_detector_xy": predictions,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
