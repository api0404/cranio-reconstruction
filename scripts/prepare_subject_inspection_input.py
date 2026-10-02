"""Recreate the masked manual-informed subject input from documented values.

The exact prior ignored transform is unavailable. Geometry here uses rounded
manual anchor_tighter values from docs/manual_auto_sc_dreg_comparison.md.
"""

import argparse
import hashlib
import json
from pathlib import Path

from canonicalize_ceph import ROOT, run


FIT_LABELS = ("S", "N", "PNS", "U1_tip", "Me")
SOURCE_SHA256 = "88b33b98bc8ac623ae54a27e65076696ef42a240d5e00289d89b3f3e279e07b3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/private/Cephalometric_X-Ray.png")
    parser.add_argument("--landmarks", type=Path, default=ROOT / "data/private/high_value_ceph_landmarks.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/subject_inspection/input")
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Documented ruler mask and rounded transform apply only to the original local ceph")
    data = json.loads(args.landmarks.read_text(encoding="utf-8"))
    points = {point["id"]: point for group in data["landmarks"].values() for point in group}
    missing = set(FIT_LABELS) - points.keys()
    if missing:
        raise ValueError(f"Missing curated manual landmarks: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "source": str(args.source.resolve()),
        "orientation": {"flip_horizontal": False, "flip_vertical": False},
        "intensity": {"mode": "identity"},
        "artifact_masks": [{"name": "vertical_ruler",
                            "polygon_original": [[2618, 30], [2710, 30], [2710, 605], [2618, 605]],
                            "fill_value": 0,
                            "non_anatomical_reason": "Reviewed ruler body in dark background of this same ceph"}],
        "landmarks": [{"name": label, "role": "check",
                       "source": [points[label]["x"], points[label]["y"]]} for label in FIT_LABELS],
        "candidates": [{"name": "manual_masked_anchor_tighter_recreated", "mode": "explicit_fov",
                        "center_oriented": [1799.64, 1099.68],
                        "fov_width_px": 1603.57, "rotation_degrees": 1.0502}],
        "provenance": {"status": "recreated_from_rounded_manual_anchor_tighter_report_not_exact_prior_input",
                       "geometry_report": "docs/manual_auto_sc_dreg_comparison.md",
                       "mask_report": "docs/sc_dreg_input_sensitivity.md",
                       "manual_landmark_file": str(args.landmarks.resolve()),
                       "landmarks_are_checks_geometry_was_fit_in_prior_experiment": True,
                       "same_source_image_mask_only": True}}
    path = args.output_dir / "generated_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    run(path, args.output_dir)


if __name__ == "__main__":
    main()
