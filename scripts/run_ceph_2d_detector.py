"""Probe a pretrained 2D ceph detector against an unchanged manual export.

Detector inputs are separate hypotheses. The network's own 800x640 resize is
recorded as a coordinate transform; it is not used for SC-DREG canonicalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from canonicalize_ceph import ROOT, read_grayscale
from ceph_canonicalization import map_points, pillow_inverse_coefficients
from pyceph_worker import PUBLISHED_CHECKPOINT_SHA256

DEFAULT_SOURCE = ROOT / "data/private/Cephalometric_X-Ray.png"
DEFAULT_MANUAL = ROOT / "data/private/high_value_ceph_landmarks.json"
DEFAULT_VENDOR = ROOT / "vendor/py-ceph"
DEFAULT_WORKER_PYTHON = ROOT / ".venv-2d/Scripts/python.exe"
PUBLISHED_PYCEPH_REVISION = "9c2f7edfc8783c2621e61d1feada764bdc3c41fe"

LABEL_MAP = {
    "Sella": "S", "Nasion": "N", "Orbitale": "Or", "Porion": "Po",
    "Subspinale": "A", "Supramentale": "B", "Pogonion": "Pog",
    "Menton": "Me", "Gonion": "Go", "Incision Inferius": "L1_tip",
    "Incision Superius": "U1_tip", "Posterior Nasal Spine": "PNS",
    "Anterior Nasal Spine": "ANS", "Articulare": "Ar",
    "Upper Lip": "Ls", "Lower Lip": "Li", "Subnasale": "Sn",
    "Soft Tissue Pogonion": "Pog_soft",
}
PRIMARY_LABELS = {"S", "N", "Or", "Po", "A", "B", "Pog", "Me",
                  "L1_tip", "U1_tip", "PNS", "ANS", "Ar"}
NETWORK_SIZE = (640, 800)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def center_scale_matrix(source_size: tuple[int, int], scale_x: float,
                        scale_y: float, center_source: tuple[float, float]) -> np.ndarray:
    matrix = np.array([[scale_x, 0, 319.5 - scale_x * center_source[0]],
                       [0, scale_y, 399.5 - scale_y * center_source[1]],
                       [0, 0, 1]], dtype=np.float64)
    if not np.isfinite(matrix).all() or np.linalg.det(matrix) == 0:
        raise ValueError(f"Invalid detector transform for {source_size}")
    return matrix


def detector_inputs(original: Image.Image, output_dir: Path) -> dict[str, dict]:
    """Three explicit detector-only FOV hypotheses, none using manual marks."""
    width, height = original.size
    if width < 2 or height < 2:
        raise ValueError("Source image is too small")
    center = ((width - 1) / 2, (height - 1) / 2)
    full_matrix = center_scale_matrix(original.size, 640 / width, 800 / height, center)
    fit_matrix = center_scale_matrix(original.size, 640 / width, 640 / width, center)
    crop_scale = 800 / height
    crop_center = (width - .5 - 640 / crop_scale / 2, (height - 1) / 2)
    crop_matrix = center_scale_matrix(original.size, crop_scale, crop_scale, crop_center)
    plans = {
        "full_stretch": (full_matrix, "Upstream full-raster resize to 800x640; nonuniform if aspect differs", None),
        "full_letterbox": (fit_matrix, "Uniform full-raster scale with vertical padding", NETWORK_SIZE),
        "right_portrait_crop": (crop_matrix, "Uniform portrait FOV anchored at source right border", NETWORK_SIZE),
    }
    inputs = {}
    for name, (matrix, description, raster_size) in plans.items():
        path = output_dir / f"{name}.png"
        if raster_size is None:
            rendered = original
        else:
            rendered = original.transform(raster_size, Image.Transform.AFFINE,
                                          pillow_inverse_coefficients(matrix),
                                          resample=Image.Resampling.BICUBIC, fillcolor=0)
        rendered.convert("RGB").save(path)
        inverse = np.linalg.inv(matrix)
        corners = map_points(inverse, [[-.5, -.5], [639.5, -.5],
                                       [639.5, 799.5], [-.5, 799.5]])
        inputs[name] = {
            "path": str(path.resolve()), "description": description,
            "detector_input_size_wh": list(rendered.size),
            "original_to_detector": matrix.tolist(),
            "detector_to_original": inverse.tolist(),
            "source_fov_corners": corners.tolist(),
            "linear_singular_values": np.linalg.svd(matrix[:2, :2], compute_uv=False).tolist(),
        }
    return inputs


def angle_at(vertex: np.ndarray, point_a: np.ndarray, point_b: np.ndarray) -> float:
    a, b = point_a - vertex, point_b - vertex
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        raise ValueError("Coincident points in cephalometric angle")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1, 1))))


def ceph_angles(points: dict[str, list[float]]) -> dict[str, float]:
    if not {"S", "N", "A", "B"}.issubset(points):
        return {}
    p = {key: np.asarray(value) for key, value in points.items()}
    sna = angle_at(p["N"], p["S"], p["A"])
    snb = angle_at(p["N"], p["S"], p["B"])
    return {"SNA_degrees": sna, "SNB_degrees": snb, "ANB_degrees": sna - snb}


def summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean_px": float(np.mean(values)),
            "median_px": float(np.median(values)), "max_px": float(np.max(values)),
            "rmse_px": float(np.sqrt(np.mean(np.square(values))))}


def draw_overlay(original: Image.Image, reference: dict, predictions: dict,
                 output_path: Path) -> None:
    canvas = original.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for label, point in predictions.items():
        if label not in reference:
            continue
        px, py = point
        mx, my = reference[label]
        draw.line((px, py, mx, my), fill=(255, 180, 0), width=4)
        draw.ellipse((px - 14, py - 14, px + 14, py + 14), outline=(255, 230, 0), width=5)
        draw.line((mx - 16, my - 16, mx + 16, my + 16), fill=(0, 240, 255), width=5)
        draw.line((mx - 16, my + 16, mx + 16, my - 16), fill=(0, 240, 255), width=5)
        draw.text((px + 18, py - 14), label, fill=(255, 230, 0))
    canvas.save(output_path)


def run(source_path: Path, manual_path: Path, output_dir: Path,
        vendor_root: Path, worker_python: Path) -> dict:
    if not source_path.is_file() or not manual_path.is_file():
        raise FileNotFoundError("Original ceph and curated manual JSON must exist")
    if not worker_python.is_file() or not (vendor_root / "src/pyceph/pyceph.py").is_file():
        raise FileNotFoundError("Optional py-ceph checkout or .venv-2d is absent")
    checkpoint = vendor_root / "src/pyceph/pretrained_models/12-26-22.pkl.gz"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Published py-ceph checkpoint missing: {checkpoint}")
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != PUBLISHED_CHECKPOINT_SHA256:
        raise ValueError("Published py-ceph checkpoint SHA-256 mismatch; refusing pickle load")
    revision = subprocess.run(["git", "-C", str(vendor_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    if revision != PUBLISHED_PYCEPH_REVISION:
        raise ValueError(f"py-ceph checkout revision {revision} differs from pinned {PUBLISHED_PYCEPH_REVISION}")
    original, source_metadata = read_grayscale(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = detector_inputs(original, output_dir)
    manifest = output_dir / "worker_inputs.json"
    worker_output = output_dir / "worker_predictions.json"
    manifest.write_text(json.dumps({name: item["path"] for name, item in plans.items()}, indent=2) + "\n")
    command = [str(worker_python), str(ROOT / "scripts/pyceph_worker.py"),
               str(manifest), str(worker_output), "--pyceph-root", str(vendor_root)]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (output_dir / "worker.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode or not worker_output.is_file():
        raise RuntimeError(f"py-ceph worker failed; see {output_dir / 'worker.log'}")
    worker = json.loads(worker_output.read_text(encoding="utf-8"))

    # Ground truth enters only after every detector prediction has been saved.
    manual_hash = sha256(manual_path)
    curated = json.loads(manual_path.read_text(encoding="utf-8"))
    if curated["coordinate_system"]["units"] != "source_image_pixels":
        raise ValueError("Curated manual JSON must use original source-image pixels")
    reference = {item["id"]: [float(item["x"]), float(item["y"])]
                 for group in curated["landmarks"].values() for item in group}
    report = {
        "status": "single_subject_detector_comparison_not_clinical_validation",
        "source": source_metadata,
        "manual_reference_path": str(manual_path.resolve()),
        "manual_reference_sha256": manual_hash,
        "pyceph_revision": revision,
        "checkpoint_sha256": checkpoint_hash,
        "worker": {key: value for key, value in worker.items() if key != "predictions_detector_xy"},
        "primary_evaluation_labels": sorted(PRIMARY_LABELS),
        "unsupported_manual_labels": sorted(set(reference) - set(LABEL_MAP.values())),
        "manual_measurements_derived_from_points": ceph_angles(reference),
        "strategies": {},
    }
    for strategy, plan in plans.items():
        raw = worker["predictions_detector_xy"][strategy]
        matrix = np.asarray(plan["detector_to_original"], dtype=np.float64)
        predictions = {LABEL_MAP[name]: map_points(matrix, [xy])[0].tolist()
                       for name, xy in raw.items() if name in LABEL_MAP}
        errors = {label: {"error_px": float(np.linalg.norm(np.asarray(point) - reference[label])),
                          "predicted_source_xy": point, "manual_source_xy": reference[label]}
                  for label, point in predictions.items() if label in reference}
        primary_errors = [item["error_px"] for label, item in errors.items() if label in PRIMARY_LABELS]
        plan["predictions_detector_xy"] = raw
        plan["predictions_source_xy"] = predictions
        plan["errors_vs_manual"] = errors
        plan["primary_summary"] = summary(primary_errors)
        plan["all_supported_summary"] = summary([v["error_px"] for v in errors.values()])
        plan["automatic_measurements_derived_from_points"] = ceph_angles(predictions)
        plan["ground_truth_used_for_detector_input"] = False
        report["strategies"][strategy] = plan
        automatic = {
            "schema_version": 1, "kind": "automatic_pyceph_unreviewed",
            "source_image_sha256": source_metadata["sha256"],
            "coordinate_system": curated["coordinate_system"],
            "strategy": strategy, "landmarks": {key: {"x": xy[0], "y": xy[1]}
                                         for key, xy in predictions.items()},
            "warning": "Never replace the curated manual export with this prediction file",
        }
        (output_dir / f"automatic_{strategy}.json").write_text(json.dumps(automatic, indent=2) + "\n")
        draw_overlay(original, reference, predictions, output_dir / f"overlay_{strategy}.png")
    if sha256(manual_path) != manual_hash:
        raise RuntimeError("Manual reference changed during evaluation")
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ceph_2d_detector")
    parser.add_argument("--pyceph-root", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    args = parser.parse_args()
    report = run(args.source, args.manual, args.output_dir, args.pyceph_root, args.worker_python)
    for name, item in report["strategies"].items():
        score = item["primary_summary"]
        print(f"{name}: n={score['count']} mean={score['mean_px']:.1f}px "
              f"median={score['median_px']:.1f}px max={score['max_px']:.1f}px")
    print(args.output_dir / "evaluation.json")


if __name__ == "__main__":
    main()
