"""Make reviewable 128x128 SC-DREG input candidates from a clinical ceph.

The coordinate map is reversible; the raster crop/downsampling is not.  The
unaltered original is retained at its source path and identified by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import PIL
from PIL import Image, ImageDraw

from ceph_canonicalization import (MODEL_CENTER, fit_similarity, fov_transform,
                                    geometry_diagnostics, map_points,
                                    orientation_matrix, pillow_inverse_coefficients,
                                    scale_about_model_center)


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_grayscale(path: Path) -> tuple[Image.Image, dict]:
    with Image.open(path) as original:
        original.load()
        if original.mode in ("RGBA", "LA"):
            alpha = original.getchannel("A")
            if alpha.getextrema() != (255, 255):
                raise ValueError("Transparent pixels require an explicit compositing decision")
        if original.mode in ("RGB", "RGBA"):
            rgb = np.asarray(original.convert("RGB"))
            if not np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) or not np.array_equal(rgb[:, :, 0], rgb[:, :, 2]):
                raise ValueError("Color channels differ; provide a deliberate grayscale export")
        if original.mode not in ("L", "RGB", "RGBA", "LA"):
            raise ValueError(f"Unsupported source mode {original.mode}; export a grayscale PNG")
        metadata = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size": list(original.size), "mode": original.mode,
                    "embedded_metadata": {k: str(v) for k, v in original.info.items()}}
        return original.convert("L"), metadata


def intensity_map(image: Image.Image, config: dict) -> tuple[Image.Image, dict]:
    mode = config.get("mode", "identity")
    data = np.asarray(image, dtype=np.float64)
    if mode == "identity":
        return image, {"mode": mode, "gain": 1.0, "offset": 0.0, "clipped_fraction": 0.0}
    if mode == "invert":
        return Image.fromarray((255 - data).astype(np.uint8), "L"), {
            "mode": mode, "gain": -1.0, "offset": 255.0, "clipped_fraction": 0.0}
    if mode != "affine":
        raise ValueError(f"Unsupported intensity mode: {mode}")
    gain, offset = float(config["gain"]), float(config.get("offset", 0.0))
    if not np.isfinite(gain) or gain == 0 or not np.isfinite(offset):
        raise ValueError("Intensity gain must be nonzero and gain/offset finite")
    mapped = gain * data + offset
    clipped = np.mean((mapped < 0) | (mapped > 255))
    return Image.fromarray(np.rint(np.clip(mapped, 0, 255)).astype(np.uint8), "L"), {
        "mode": mode, "gain": gain, "offset": offset, "clipped_fraction": float(clipped),
        "warning": "Clipping and uint8 quantization are not intensity-invertible; original is retained"}


def draw_points(draw: ImageDraw.ImageDraw, items: list, matrix: np.ndarray,
                key: str, scale: float = 1.0, offset=(0, 0)) -> None:
    for item in items:
        if key not in item:
            continue
        point = map_points(matrix, [item[key]])[0] * scale + offset
        x, y = point
        color = (255, 70, 70) if item.get("role", "fit") == "fit" else (20, 220, 255)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
        draw.text((x + 5, y - 8), item["name"], fill=color)


def make_preview(original: Image.Image, model: Image.Image, target: Image.Image | None,
                 matrix: np.ndarray, landmarks: list) -> Image.Image:
    source_scale = min(900 / original.width, 750 / original.height)
    source_preview = original.resize((round(original.width * source_scale),
                                      round(original.height * source_scale)), Image.Resampling.LANCZOS).convert("RGB")
    right_width = 512 + (528 if target is not None else 0)
    sheet = Image.new("RGB", (source_preview.width + right_width + 32,
                              max(source_preview.height, 540) + 30), (20, 20, 20))
    sheet.paste(source_preview, (8, 25))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), "Original ceph / selected FOV", fill="white")
    corners = map_points(np.linalg.inv(matrix), [[-0.5, -0.5], [127.5, -0.5],
                                                  [127.5, 127.5], [-0.5, 127.5]])
    polygon = [(float(x * source_scale + 8), float(y * source_scale + 25)) for x, y in corners]
    draw.line(polygon + polygon[:1], fill=(255, 230, 0), width=3)
    draw_points(draw, landmarks, np.eye(3), "source", source_scale, (8, 25))
    model_large = model.resize((512, 512), Image.Resampling.NEAREST).convert("RGB")
    x_model = source_preview.width + 16
    sheet.paste(model_large, (x_model, 25))
    draw.text((x_model, 6), "128x128 candidate (8x preview)", fill="white")
    draw_points(draw, landmarks, matrix, "source", 4, (x_model, 25))
    if target is not None:
        x_target = x_model + 528
        sheet.paste(target.resize((512, 512), Image.Resampling.NEAREST).convert("RGB"), (x_target, 25))
        draw.text((x_target, 6), "Published 04002 (different subject)", fill="white")
        draw_points(draw, landmarks, np.eye(3), "target", 4, (x_target, 25))
    return sheet


def candidates_from_config(config: dict, size: tuple[int, int], landmarks: list) -> list[tuple[str, np.ndarray, dict]]:
    orientation = config.get("orientation", {})
    orient = orientation_matrix(*size, bool(orientation.get("flip_horizontal", False)),
                                bool(orientation.get("flip_vertical", False)))
    candidates = config.get("candidates")
    if candidates is None:
        candidates = [{"name": "landmark_fit", "mode": "landmark_similarity", "scale_factor": 1},
                      {"name": "wider_fov", "mode": "landmark_similarity", "scale_factor": 0.85},
                      {"name": "tighter_fov", "mode": "landmark_similarity", "scale_factor": 1.15}]
    if not candidates:
        raise ValueError("At least one candidate is required")
    result = []
    used = set()
    for candidate in candidates:
        name, mode = candidate["name"], candidate["mode"]
        if not name.isidentifier() or name in used:
            raise ValueError("Candidate names must be unique identifiers")
        used.add(name)
        if mode == "landmark_similarity":
            fit = [item for item in landmarks if item.get("role", "fit") == "fit"]
            source = map_points(orient, [item["source"] for item in fit])
            target = np.asarray([item["target"] for item in fit], dtype=np.float64)
            weights = [item.get("weight", 1.0) for item in fit]
            base = fit_similarity(source, target, weights) @ orient
            matrix = scale_about_model_center(base, float(candidate.get("scale_factor", 1)))
            details = {"mode": mode, "fit_landmarks": [item["name"] for item in fit],
                       "scale_factor_vs_fit": float(candidate.get("scale_factor", 1))}
        elif mode == "explicit_fov":
            matrix = fov_transform(candidate["center_oriented"], float(candidate["fov_width_px"]),
                                   float(candidate.get("rotation_degrees", 0)), orient)
            details = {"mode": mode, "center_oriented": candidate["center_oriented"],
                       "fov_width_px": candidate["fov_width_px"],
                       "rotation_degrees": candidate.get("rotation_degrees", 0)}
        else:
            raise ValueError(f"Unsupported candidate mode: {mode}")
        result.append((name, matrix, details))
    return result


def run(config_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    source_path = resolve(config["source"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Source ceph missing: {source_path}")
    original, source_metadata = read_grayscale(source_path)
    target = None
    target_metadata = None
    if "target" in config:
        target_path = resolve(config["target"])
        with Image.open(target_path) as loaded:
            if loaded.size != (128, 128):
                raise ValueError("Reference target must be 128x128")
            target = loaded.convert("L")
        target_metadata = {"path": str(target_path.resolve()),
                           "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest()}
    landmarks = config.get("landmarks", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, matrix, details in candidates_from_config(config, original.size, landmarks):
        diagnostics = geometry_diagnostics(matrix, original.size)
        candidate = original.transform((128, 128), Image.Transform.AFFINE,
                                       pillow_inverse_coefficients(matrix),
                                       resample=Image.Resampling.BICUBIC, fillcolor=0)
        candidate, intensity = intensity_map(candidate, config.get("intensity", {"mode": "identity"}))
        candidate.save(output_dir / f"{name}_128.png")
        make_preview(original, candidate, target, matrix, landmarks).save(output_dir / f"{name}_preview.png")
        mapped = []
        for item in landmarks:
            predicted = map_points(matrix, [item["source"]])[0]
            record = {"name": item["name"], "role": item.get("role", "fit"),
                      "original": item["source"], "model_predicted": predicted.tolist(),
                      "round_trip_original": map_points(np.linalg.inv(matrix), [predicted])[0].tolist()}
            if "target" in item:
                record["template_target"] = item["target"]
                record["template_residual_px"] = float(np.linalg.norm(predicted - item["target"]))
            mapped.append(record)
        reports.append({"name": name, **details, "original_to_model": matrix.tolist(),
                        "model_to_original": np.linalg.inv(matrix).tolist(),
                        "pillow_inverse_coefficients": pillow_inverse_coefficients(matrix),
                        "geometry": diagnostics, "intensity": intensity, "landmarks": mapped,
                        "model_image": f"{name}_128.png", "preview": f"{name}_preview.png"})
        print(f"{name}: FOV={diagnostics['source_fov_width_px']:.1f} source px; "
              f"inside={diagnostics['all_fov_corners_inside_source']}; "
              f"preview={output_dir / (name + '_preview.png')}")
    report = {"status": "candidate_clinical_input_not_validated", "source": source_metadata,
              "target": target_metadata, "pillow_version": PIL.__version__,
              "config": str(config_path.resolve()), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "coordinate_convention": "pixel centers; x right, y down; origin at top-left pixel center",
              "spatial_warning": "Coordinate transform is invertible; cropped/resampled raster is not",
              "orientation": config.get("orientation", {}),
              "resampling": "Pillow bicubic, one spatial resampling", "candidates": reports}
    (output_dir / "transforms.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Local JSON with source, candidate definitions and optional landmarks")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ceph_canonicalization")
    args = parser.parse_args()
    try:
        run(args.config, args.output_dir)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"Canonicalization failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
