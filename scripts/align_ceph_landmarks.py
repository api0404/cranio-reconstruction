"""Exploratory 2D landmark similarity alignment to the SC-DREG example frame.

This is separate from the published inference baseline. Landmark coordinates
must be supplied explicitly; no clinical landmark detector is implied.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def fit_similarity(source: np.ndarray, target: np.ndarray, weights: np.ndarray):
    """Fit target = scale * source @ rotation + translation, without reflection."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("At least three corresponding 2D landmarks are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Landmark coordinates must be finite")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Landmark weights must be positive and finite")
    weights = weights / weights.sum()
    src_center = np.sum(source * weights[:, None], axis=0)
    dst_center = np.sum(target * weights[:, None], axis=0)
    src = source - src_center
    dst = target - dst_center
    covariance = (src * weights[:, None]).T @ dst
    u, singular, vt = np.linalg.svd(covariance)
    signs = np.diag([1.0, np.linalg.det(u @ vt)])
    rotation = u @ signs @ vt
    denominator = np.sum(weights * np.sum(src * src, axis=1))
    if denominator <= 1e-12:
        raise ValueError("Fit landmarks are coincident")
    scale = float(np.sum(singular * np.diag(signs)) / denominator)
    if scale <= 0:
        raise ValueError("Invalid similarity scale")
    translation = dst_center - scale * src_center @ rotation
    return scale, rotation, translation


def transform_points(points, scale, rotation, translation):
    return scale * points @ rotation + translation


def pil_inverse_affine(scale, rotation, translation):
    """Coefficients mapping each output pixel to the source pixel for PIL."""
    inverse = rotation.T / scale
    offset = -translation @ inverse
    return (float(inverse[0, 0]), float(inverse[1, 0]), float(offset[0]),
            float(inverse[0, 1]), float(inverse[1, 1]), float(offset[1]))


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def draw_landmarks(image: Image.Image, landmarks: list, key: str) -> Image.Image:
    marked = image.convert("RGB")
    draw = ImageDraw.Draw(marked)
    for item in landmarks:
        x, y = item[key]
        color = (255, 55, 55) if item.get("role", "fit") == "fit" else (50, 220, 255)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color, width=1)
        draw.text((x + 3, y - 6), item["name"], fill=color)
    return marked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Local JSON with source/target paths and corresponding landmarks")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "landmark_alignment")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path, target_path = resolve(config["source"]), resolve(config["target"])
    source_image = Image.open(source_path).convert("L")
    target_image = Image.open(target_path).convert("L")
    if target_image.size != (128, 128):
        raise ValueError(f"Target frame must be 128x128, got {target_image.size}")
    landmarks = config["landmarks"]
    fit = [item for item in landmarks if item.get("role", "fit") == "fit"]
    src = np.asarray([item["source"] for item in fit], dtype=np.float64)
    dst = np.asarray([item["target"] for item in fit], dtype=np.float64)
    weights = np.asarray([item.get("weight", 1.0) for item in fit], dtype=np.float64)
    scale, rotation, translation = fit_similarity(src, dst, weights)
    angle = float(np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0])))
    aligned = source_image.transform((128, 128), Image.Transform.AFFINE,
                                     pil_inverse_affine(scale, rotation, translation),
                                     resample=Image.Resampling.BICUBIC, fillcolor=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned.save(args.output_dir / "aligned_128.png")
    Image.blend(aligned, target_image, 0.5).save(args.output_dir / "overlay_128.png")
    target_marked = draw_landmarks(target_image, landmarks, "target")
    target_marked.resize((768, 768), Image.Resampling.NEAREST).save(args.output_dir / "target_landmarks.png")
    # Mark the transformed source-point positions on the aligned preview.
    transformed = []
    report_points = []
    for item in landmarks:
        predicted = transform_points(np.asarray([item["source"]], dtype=np.float64),
                                     scale, rotation, translation)[0]
        error = float(np.linalg.norm(predicted - np.asarray(item["target"], dtype=np.float64)))
        transformed.append({**item, "predicted": predicted.tolist()})
        report_points.append({"name": item["name"], "role": item.get("role", "fit"),
                              "source": item["source"], "target": item["target"],
                              "predicted": predicted.tolist(), "residual_px": error})
    aligned_marked = draw_landmarks(aligned, transformed, "predicted")
    aligned_marked.resize((768, 768), Image.Resampling.NEAREST).save(args.output_dir / "aligned_landmarks.png")
    report = {"status": "provisional_visual_landmarks", "source": str(source_path),
              "target": str(target_path), "scale_target_px_per_source_px": scale,
              "rotation_degrees_image_coordinates": angle,
              "rotation_row_matrix": rotation.tolist(), "translation_px": translation.tolist(),
              "inverse_affine_for_pil": pil_inverse_affine(scale, rotation, translation),
              "resampling": "Pillow bicubic", "landmarks": report_points}
    (args.output_dir / "fit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"scale={scale:.8f} target px/source px, rotation={angle:.4f} degrees")
    for point in report_points:
        print(f"{point['name']} ({point['role']}): residual={point['residual_px']:.2f} target pixels")
    print(f"Preview and fit saved under {args.output_dir}")


if __name__ == "__main__":
    main()
