"""Fit a similarity-only SC-DREG input from manual or unreviewed 2D marks.

The template is explicitly case-specific: curated points mapped through a
previous experimental transform. It is not a published SC-DREG normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from canonicalize_ceph import ROOT, read_grayscale, run as canonicalize
from ceph_canonicalization import map_points
from run_ceph_2d_detector import sha256

DEFAULT_SOURCE = ROOT / "data/private/Cephalometric_X-Ray.png"
DEFAULT_MANUAL = ROOT / "data/private/high_value_ceph_landmarks.json"
DEFAULT_TRANSFORMS = ROOT / "outputs/sc_dreg_sensitivity/inputs/transforms.json"
FIT_LABELS = ("S", "N", "PNS", "U1_tip", "Me")


def read_landmarks(path: Path, source_hash: str) -> tuple[dict[str, list[float]], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["coordinate_system"]["units"] != "source_image_pixels":
        raise ValueError("Landmarks must use original source-image pixels")
    if data.get("kind") == "automatic_pyceph_unreviewed":
        if data["source_image_sha256"] != source_hash:
            raise ValueError("Automatic predictions belong to a different source image")
        points = {key: [float(value["x"]), float(value["y"])]
                  for key, value in data["landmarks"].items()}
        kind = "automatic_pyceph_unreviewed"
    else:
        if "source_image_sha256" in data and data["source_image_sha256"] != source_hash:
            raise ValueError("Manual landmarks belong to a different source image")
        points = {item["id"]: [float(item["x"]), float(item["y"])]
                  for group in data["landmarks"].values() for item in group}
        kind = "curated_manual"
    if not all(np.isfinite(xy).all() for xy in points.values()):
        raise ValueError("Landmark coordinates must be finite")
    return points, kind


def build_config(source: Path, marks_path: Path, manual_path: Path,
                 transforms_path: Path, mask_config_path: Path | None = None) -> dict:
    original, source_meta = read_grayscale(source)
    source_hash = source_meta["sha256"]
    marks, kind = read_landmarks(marks_path, source_hash)
    manual, manual_kind = read_landmarks(manual_path, source_hash)
    if manual_kind != "curated_manual":
        raise ValueError("Template source must be the curated manual export")
    report = json.loads(transforms_path.read_text(encoding="utf-8"))
    if report["source"]["sha256"] != source_hash:
        raise ValueError("Experimental template transform belongs to a different source image")
    reference = next((item for item in report["candidates"] if item["name"] == "tighter_fov"), None)
    if reference is None:
        raise ValueError("Experimental transforms lack tighter_fov candidate")
    template_matrix = np.asarray(reference["original_to_model"], dtype=np.float64)
    if template_matrix.shape != (3, 3) or not np.isfinite(template_matrix).all():
        raise ValueError("Invalid template matrix")
    missing = set(FIT_LABELS) - marks.keys() | (set(FIT_LABELS) - manual.keys())
    if missing:
        raise ValueError(f"Missing fit landmarks: {sorted(missing)}")
    landmarks = []
    for label in FIT_LABELS:
        source_xy = marks[label]
        if not 0 <= source_xy[0] < original.width or not 0 <= source_xy[1] < original.height:
            raise ValueError(f"{label} lies outside the original image")
        target = map_points(template_matrix, [manual[label]])[0].tolist()
        landmarks.append({"name": label, "role": "fit", "source": source_xy, "target": target})
    masks = []
    if mask_config_path is not None:
        masks = json.loads(mask_config_path.read_text(encoding="utf-8")).get("artifact_masks", [])
    return {
        "source": str(source.resolve()),
        "target": str((ROOT / "vendor/sc-dreg/tests/04002.png").resolve()),
        "orientation": {"flip_horizontal": False, "flip_vertical": False},
        "intensity": {"mode": "identity"},
        "artifact_masks": masks,
        "landmarks": landmarks,
        "candidates": [
            {"name": "anchor_fit", "mode": "landmark_similarity", "scale_factor": 1},
            {"name": "anchor_wider", "mode": "landmark_similarity", "scale_factor": 0.85},
            {"name": "anchor_tighter", "mode": "landmark_similarity", "scale_factor": 1.15},
        ],
        "provenance": {
            "status": "case_specific_experimental_template_not_published_training_geometry",
            "landmark_kind": kind, "landmark_file": str(marks_path.resolve()),
            "landmark_sha256": sha256(marks_path),
            "manual_template_file": str(manual_path.resolve()),
            "manual_template_sha256": sha256(manual_path),
            "template_transform_file": str(transforms_path.resolve()),
            "template_transform_sha256": sha256(transforms_path),
            "template_candidate": "tighter_fov",
            "fit_labels": list(FIT_LABELS),
            "template_not_independent_ground_truth": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--manual-template", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--template-transforms", type=Path, default=DEFAULT_TRANSFORMS)
    parser.add_argument("--mask-config", type=Path, default=ROOT / "data/private/sc_dreg_sensitivity.json")
    parser.add_argument("--output-dir", type=Path, help="Ignored candidate directory; defaults by landmark kind")
    args = parser.parse_args()
    config = build_config(args.source, args.landmarks, args.manual_template,
                          args.template_transforms, args.mask_config)
    output_dir = args.output_dir
    if output_dir is None:
        suffix = ("manual" if config["provenance"]["landmark_kind"] == "curated_manual"
                  else args.landmarks.stem)
        output_dir = ROOT / "outputs/ceph_landmark_canonicalization" / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "generated_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    canonicalize(config_path, output_dir)
    print(output_dir / "transforms.json")


if __name__ == "__main__":
    main()
