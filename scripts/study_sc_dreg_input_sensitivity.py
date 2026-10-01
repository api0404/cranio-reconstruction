"""Run masked clinical-ceph candidates and a small local SC-DREG input sweep.

All outputs are exploratory and belong under ignored outputs/. No CBCT ground
truth or clinical landmark reprojection ground truth is available here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from canonicalize_ceph import run as canonicalize
from ceph_canonicalization import (MODEL_CENTER, geometry_diagnostics, map_points,
                                    pillow_inverse_coefficients, scale_about_model_center,
                                    translation)
from run_sc_dreg_demo import ROOT, validate_inputs
from sc_dreg_compat import load_model_class


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    x, y = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 0 else float("nan")


def edge_magnitude(a: np.ndarray) -> np.ndarray:
    # Sobel gradients in x and y, zero-padding avoided by edge replication.
    p = np.pad(np.asarray(a, np.float64), 1, mode="edge")
    gx = (p[:-2, 2:] + 2*p[1:-1, 2:] + p[2:, 2:]
          - p[:-2, :-2] - 2*p[1:-1, :-2] - p[2:, :-2]) / 8
    gy = (p[2:, :-2] + 2*p[2:, 1:-1] + p[2:, 2:]
          - p[:-2, :-2] - 2*p[:-2, 1:-1] - p[:-2, 2:]) / 8
    return np.hypot(gx, gy)


def image_metrics(input_image: np.ndarray, drr: np.ndarray, torch) -> dict:
    from pytorch_msssim import ssim

    a, b = np.asarray(input_image, np.float32), np.asarray(drr, np.float32)
    ea, eb = edge_magnitude(a), edge_magnitude(b)
    ma = ea > max(float(np.quantile(ea, 0.85)), 1e-8)
    mb = eb > max(float(np.quantile(eb, 0.85)), 1e-8)
    intersection = np.count_nonzero(ma & mb)
    edge_dice = 2 * intersection / (np.count_nonzero(ma) + np.count_nonzero(mb))
    return {"ncc": ncc(a, b),
            "ssim": float(ssim(torch.from_numpy(a)[None, None],
                               torch.from_numpy(b)[None, None], data_range=1, size_average=True)),
            "edge_ncc": ncc(ea, eb), "edge_top15_dice": float(edge_dice),
            "input_mean": float(a.mean()), "drr_mean": float(b.mean())}


def distribution(a: np.ndarray) -> dict:
    a = np.asarray(a, np.float32)
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "std": float(a.std()), "p05": float(np.quantile(a, .05)),
            "p50": float(np.quantile(a, .5)), "p95": float(np.quantile(a, .95))}


def vector_magnitude(a: np.ndarray) -> dict:
    norm = np.linalg.norm(np.asarray(a, np.float32), axis=-1)
    return {"mean": float(norm.mean()), "p95": float(np.quantile(norm, .95)),
            "max": float(norm.max()), "units": "model grid voxels"}


def segmented_proxy(a: np.ndarray, threshold: float = 0.5) -> dict:
    points = np.argwhere(np.asarray(a) >= threshold)
    if not len(points):
        return {"voxel_count": 0, "centroid_array_index": None}
    return {"voxel_count": int(len(points)),
            "centroid_array_index": points.mean(axis=0).tolist(),
            "bbox_min_array_index": points.min(axis=0).tolist(),
            "bbox_max_array_index": points.max(axis=0).tolist()}


def render(image: Image.Image, matrix: np.ndarray) -> Image.Image:
    return image.transform((128, 128), Image.Transform.AFFINE,
                           pillow_inverse_coefficients(matrix),
                           resample=Image.Resampling.BICUBIC, fillcolor=0)


def rotation_about_model_center(degrees: float) -> np.ndarray:
    a = np.deg2rad(degrees)
    r = np.array([[np.cos(a), -np.sin(a), 0],
                  [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=np.float64)
    return translation(MODEL_CENTER, MODEL_CENTER) @ r @ translation(-MODEL_CENTER, -MODEL_CENTER)


def sweep_transforms(base: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [("scale_minus3pct", scale_about_model_center(base, .97)),
            ("scale_plus3pct", scale_about_model_center(base, 1.03)),
            ("rotate_minus1deg", rotation_about_model_center(-1) @ base),
            ("rotate_plus1deg", rotation_about_model_center(1) @ base),
            ("x_minus2px", translation(-2, 0) @ base),
            ("x_plus2px", translation(2, 0) @ base),
            ("y_minus2px", translation(0, -2) @ base),
            ("y_plus2px", translation(0, 2) @ base)]


def save_overlay(path: Path, raw: np.ndarray, refined: np.ndarray, landmarks: list) -> None:
    x = np.rint(np.clip(raw * 255, 0, 255)).astype(np.uint8)
    y = np.rint(np.clip(refined * 255, 0, 255)).astype(np.uint8)
    color = np.stack((x, y, (x.astype(np.uint16) + y.astype(np.uint16)) // 2), axis=-1).astype(np.uint8)
    canvas = Image.fromarray(color, "RGB").resize((512, 512), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)
    for item in landmarks:
        px, py = item["model_predicted"]
        px, py = px * 4, py * 4
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline=(255, 255, 0), width=2)
        draw.text((px + 6, py - 8), item["name"], fill=(255, 255, 0))
    canvas.save(path)


def pairwise(a: dict, b: dict) -> dict:
    va, vb = a["volume"], b["volume"]
    diff = va.astype(np.float64) - vb.astype(np.float64)
    sa, sb = a["seg"], b["seg"]
    ca = np.asarray(a["seg_proxy"]["centroid_array_index"])
    cb = np.asarray(b["seg_proxy"]["centroid_array_index"])
    pa = np.asarray(a["projected_proxy"]["centroid_array_index"])
    pb = np.asarray(b["projected_proxy"]["centroid_array_index"])
    return {"volume_mae": float(np.abs(diff).mean()),
            "volume_rmse": float(np.sqrt(np.mean(diff**2))),
            "volume_ncc": ncc(va, vb),
            "seg_dice": float(2*np.count_nonzero(sa & sb)/(np.count_nonzero(sa)+np.count_nonzero(sb))),
            "seg_3d_centroid_shift_grid_voxels": float(np.linalg.norm(ca-cb)),
            "seg_projected_centroid_shift_model_px": float(np.linalg.norm(pa-pb))}


def source_landmark_shift(a: dict, b: dict) -> dict:
    pa = np.asarray([item["model_predicted"] for item in a["source_landmarks_in_model"]])
    pb = np.asarray([item["model_predicted"] for item in b["source_landmarks_in_model"]])
    distances = np.linalg.norm(pa-pb, axis=1)
    return {"mean_projected_source_landmark_shift_model_px": float(distances.mean()),
            "max_projected_source_landmark_shift_model_px": float(distances.max())}


def save_volume_difference(path: Path, base: np.ndarray, other: np.ndarray) -> None:
    """Display a mean projection plus absolute-difference max projection."""
    anatomy = np.mean(base, axis=0)
    delta = np.max(np.abs(base-other), axis=0)
    gray = np.rint(255 * np.clip(anatomy / max(np.quantile(anatomy, .99), 1e-8), 0, 1)).astype(np.uint8)
    heat = np.rint(255 * np.clip(delta / max(np.quantile(delta, .99), 1e-8), 0, 1)).astype(np.uint8)
    result = np.stack((np.maximum(gray, heat), gray // 2, gray // 2), axis=-1)
    Image.fromarray(result.astype(np.uint8), "RGB").resize((512, 512), Image.Resampling.NEAREST).save(path)


def save_overlay_montage(output_dir: Path) -> None:
    names = ("tighter_fov", "landmark_fit", "wider_fov", "x_minus2px", "tighter_unmasked")
    canvas = Image.new("RGB", (5*320, 348), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(names):
        with Image.open(output_dir / "cases" / name / "input_drr_overlay.png") as item:
            canvas.paste(item.resize((320, 320), Image.Resampling.LANCZOS), (index*320, 28))
        draw.text((index*320+6, 7), name, fill="white")
    canvas.save(output_dir / "input_drr_montage.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "data/private/sc_dreg_sensitivity.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sc_dreg_sensitivity")
    args = parser.parse_args()
    started = time.perf_counter()
    input_dir = args.output_dir / "inputs"
    canonicalize(args.config, input_dir)
    geometry = json.loads((input_dir / "transforms.json").read_text())
    inputs = []
    for item in geometry["candidates"]:
        inputs.append((item["name"], Image.open(input_dir / item["model_image"]).convert("L"),
                       np.asarray(item["original_to_model"], dtype=np.float64)))
    base = next(matrix for name, _, matrix in inputs if name == "tighter_fov")
    masked_source = Image.open(input_dir / "masked_source.png").convert("L")
    for name, matrix in sweep_transforms(base):
        inputs.append((name, render(masked_source, matrix), matrix))
    # Quantify the source-ruler mask itself with otherwise identical geometry.
    from canonicalize_ceph import read_grayscale, resolve
    original, _ = read_grayscale(resolve(json.loads(args.config.read_text(encoding="utf-8-sig"))["source"]))
    inputs.append(("tighter_unmasked", render(original, base), base))

    import torch
    import SimpleITK as sitk
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_inputs(ROOT / "models", input_dir / "tighter_fov_128.png", 60)
    model = load_model_class()(60, str(ROOT / "models")).to(device)
    checkpoint = torch.load(ROOT / "models/cbct_c2f_model_ckpt.tar", map_location=device,
                            weights_only=True, mmap=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    by_name = {}
    metrics = {}
    for name, image, matrix in inputs:
        array = np.asarray(image, np.float32) / 255
        tensor = torch.from_numpy(array.copy())[None, None].to(device)
        with torch.inference_mode():
            output = model(tensor)
        coarse_drr, refined_drr, coarse_df, refined_df, para, _, volume, _, projected_seg, _, _, seg = output
        refine = refined_drr.detach().cpu().numpy().squeeze()
        coarse = coarse_drr.detach().cpu().numpy().squeeze()
        vol = volume.detach().cpu().numpy().squeeze().copy()
        seg_array = seg.detach().cpu().numpy().squeeze() >= .5
        projected = projected_seg.detach().cpu().numpy().squeeze()
        seg_proxy = segmented_proxy(seg_array, .5)
        proj_proxy = segmented_proxy(projected, .5)
        pca = para.detach().cpu().numpy().ravel()
        landmarks = [{"name": point["name"], "model_predicted": map_points(matrix, [point["source"]])[0].tolist()}
                     for point in json.loads(args.config.read_text(encoding="utf-8-sig"))["landmarks"]]
        case_dir = args.output_dir / "cases" / name
        case_dir.mkdir(parents=True, exist_ok=True)
        image.save(case_dir / "input.png")
        Image.fromarray(np.rint(np.clip(refine*255, 0, 255)).astype(np.uint8)).save(case_dir / "refined_drr.png")
        save_overlay(case_dir / "input_drr_overlay.png", array, refine, landmarks)
        if name in ("tighter_fov", "landmark_fit", "wider_fov"):
            sitk.WriteImage(sitk.GetImageFromArray(vol), str(case_dir / "refined.nii.gz"))
            sitk.WriteImage(sitk.GetImageFromArray(seg_array.astype(np.uint8)), str(case_dir / "mandible_seg.nii.gz"))
        record = {"name": name, "geometry": geometry_diagnostics(matrix, original.size),
                  "input_refined": image_metrics(array, refine, torch),
                  "input_coarse": image_metrics(array, coarse, torch),
                  "pca": {**distribution(pca), "l2": float(np.linalg.norm(pca)),
                          "abs_p95": float(np.quantile(np.abs(pca), .95)), "count": len(pca)},
                  "coarse_displacement": vector_magnitude(coarse_df.detach().cpu().numpy()),
                  "refined_displacement": vector_magnitude(refined_df.detach().cpu().numpy()),
                  "volume": {**distribution(vol), "fraction_gt_0_55": float(np.mean(vol > .55))},
                  "seg_proxy": seg_proxy, "projected_proxy": proj_proxy,
                  "source_landmarks_in_model": landmarks,
                  "landmark_reprojection_error": None,
                  "landmark_reprojection_note": "No independent output-DRR anatomical landmark annotations"}
        metrics[name] = record
        by_name[name] = {"volume": vol, "seg": seg_array,
                         "seg_proxy": seg_proxy, "projected_proxy": proj_proxy}
        print(f"{name}: NCC={record['input_refined']['ncc']:.4f}, "
              f"SSIM={record['input_refined']['ssim']:.4f}, "
              f"edge Dice={record['input_refined']['edge_top15_dice']:.4f}", flush=True)
        del output, tensor
    comparisons = {}
    for name in metrics:
        if name != "tighter_fov":
            comparisons[f"tighter_fov__{name}"] = {
                **pairwise(by_name["tighter_fov"], by_name[name]),
                **source_landmark_shift(metrics["tighter_fov"], metrics[name])}
    for left, right in (("landmark_fit", "wider_fov"), ("landmark_fit", "tighter_fov")):
        comparisons[f"{left}__{right}"] = {**pairwise(by_name[left], by_name[right]),
                                           **source_landmark_shift(metrics[left], metrics[right])}
    for name in ("landmark_fit", "wider_fov", "x_minus2px", "y_minus2px"):
        save_volume_difference(args.output_dir / f"volume_difference_tighter_vs_{name}.png",
                               by_name["tighter_fov"]["volume"], by_name[name]["volume"])
    save_overlay_montage(args.output_dir)
    report = {"status": "input_domain_sensitivity_without_cbct_ground_truth",
              "landmarks_status": "rough_visual_regions_not_reviewed_cephalometric_points",
              "device": str(device), "torch": torch.__version__,
              "input_config_sha256": geometry["config_sha256"],
              "source_sha256": geometry["source"]["sha256"],
              "artifact_masks": geometry["artifact_masks"],
              "runtime_seconds": time.perf_counter()-started,
              "cases": metrics, "pairwise": comparisons}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "study.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"Study saved to {args.output_dir / 'study.json'}")


if __name__ == "__main__":
    main()
