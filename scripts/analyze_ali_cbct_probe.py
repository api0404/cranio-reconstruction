"""Project ALI probe points through SC-DREG and optional ceph transform metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ali_cbct_probe import project_drr_xyz


def map_xy(matrix: np.ndarray, xy: tuple[float, float]) -> list[float]:
    value = matrix @ np.array([xy[0], xy[1], 1.0])
    return (value[:2] / value[2]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_json", type=Path)
    parser.add_argument("image_128", type=Path)
    parser.add_argument("--transforms", type=Path, help="canonicalization transforms.json")
    parser.add_argument("--candidate", default="tighter_fov")
    parser.add_argument("--curated-landmarks", type=Path, help="private source-pixel landmark JSON")
    parser.add_argument("--original-image", type=Path, help="optional private full-resolution ceph overlay")
    args = parser.parse_args()

    report = json.loads(args.probe_json.read_text(encoding="utf-8"))
    predicted = report["model_xyz_continuous_index"]
    projected = {name: list(project_drr_xyz(xyz)) for name, xyz in predicted.items()}
    output = {"projection_model_uv": projected,
              "projection_convention": "SC-DREG GenerateDRR grid, align_corners=False, then 180-degree rotation",
              "physical_scale_warning": "All coordinates are pixels or uncalibrated SC-DREG indices, not mm."}
    targets = {}
    if args.transforms and args.curated_landmarks:
        transform_data = json.loads(args.transforms.read_text(encoding="utf-8"))
        candidate = next((x for x in transform_data["candidates"] if x["name"] == args.candidate), None)
        if candidate is None:
            parser.error(f"Candidate {args.candidate} is absent from {args.transforms}")
        original_to_model = np.asarray(candidate["original_to_model"], dtype=float)
        model_to_original = np.linalg.inv(original_to_model)
        source = json.loads(args.curated_landmarks.read_text(encoding="utf-8"))
        target_source = {x["id"]: [float(x["x"]), float(x["y"])]
                         for x in source["landmarks"]["skeletal"]}
        comparisons = {}
        for name, uv in projected.items():
            if name not in target_source:
                continue
            target_uv = map_xy(original_to_model, target_source[name])
            source_predicted = map_xy(model_to_original, uv)
            comparisons[name] = {
                "target_source_xy": target_source[name],
                "target_model_uv": target_uv,
                "prediction_model_uv": uv,
                "prediction_source_xy": source_predicted,
                "error_model_pixels": float(np.linalg.norm(np.asarray(uv) - target_uv)),
                "error_source_pixels": float(np.linalg.norm(np.asarray(source_predicted) - target_source[name])),
            }
            targets[name] = target_uv
        output["candidate"] = args.candidate
        output["comparisons"] = comparisons
        output["meaning"] = "ALI-vs-curated-2D reprojection discrepancy; not 3D anatomical accuracy"
    elif bool(args.transforms) != bool(args.curated_landmarks):
        parser.error("Provide both --transforms and --curated-landmarks")

    image = Image.open(args.image_128).convert("RGB")
    if image.size != (128, 128):
        parser.error(f"Expected a 128x128 model input, got {image.size}")
    image = image.resize((768, 768), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    for name, (u, v) in projected.items():
        x, y = 6 * u, 6 * v
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="yellow", width=3)
        draw.text((x + 10, y - 9), name, fill="yellow")
    for name, (u, v) in targets.items():
        x, y = 6 * u, 6 * v
        draw.line((x - 9, y - 9, x + 9, y + 9), fill="cyan", width=3)
        draw.line((x - 9, y + 9, x + 9, y - 9), fill="cyan", width=3)
        draw.text((x + 10, y + 7), f"{name} target", fill="cyan")
    destination = args.probe_json.parent
    image.save(destination / "projection_overlay.png")
    if args.original_image:
        if not output.get("comparisons"):
            parser.error("--original-image requires the transform and curated landmarks")
        original = Image.open(args.original_image).convert("RGB")
        original_draw = ImageDraw.Draw(original)
        for name, item in output["comparisons"].items():
            x, y = item["prediction_source_xy"]
            original_draw.ellipse((x - 18, y - 18, x + 18, y + 18), outline="yellow", width=6)
            original_draw.text((x + 23, y - 18), name, fill="yellow")
            x, y = item["target_source_xy"]
            original_draw.line((x - 22, y - 22, x + 22, y + 22), fill="cyan", width=6)
            original_draw.line((x - 22, y + 22, x + 22, y - 22), fill="cyan", width=6)
            original_draw.text((x + 23, y + 12), f"{name} target", fill="cyan")
        original.save(destination / "projection_original_overlay.png")
    (destination / "projection_analysis.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination / "projection_analysis.json")


if __name__ == "__main__":
    main()
