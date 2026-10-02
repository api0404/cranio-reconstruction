"""Compare manual- and py-ceph-driven anchor_tighter SC-DREG inference.

Both forward passes use the published runner's preprocessing, checkpoint,
training-mode model and compatibility wrapper. All outputs stay ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from canonicalize_ceph import ROOT
from ceph_canonicalization import fit_similarity, map_points
from ali_cbct_probe import project_drr_xyz
from run_sc_dreg_demo import validate_inputs
from sc_dreg_compat import load_model_class
from study_sc_dreg_input_sensitivity import ncc

BASE = ROOT / "outputs/manual_vs_pyceph"
CANON = ROOT / "outputs/ceph_landmark_canonicalization"
INPUTS = {
    "manual": CANON / "manual/anchor_tighter_128.png",
    "automatic": CANON / "automatic_full_stretch/anchor_tighter_128.png",
}
TRANSFORMS = {key: path.parent / "transforms.json" for key, path in INPUTS.items()}
MANUAL_JSON = ROOT / "data/private/high_value_ceph_landmarks.json"
DETECTOR_REPORT = ROOT / "outputs/ceph_2d_detector/evaluation.json"


def pair_metrics(a, b) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = b - a
    return {"shape": list(a.shape), "manual_min": float(a.min()), "manual_max": float(a.max()),
            "automatic_min": float(b.min()), "automatic_max": float(b.max()),
            "mae": float(np.mean(np.abs(d))), "rmse": float(np.sqrt(np.mean(d*d))),
            "max_abs": float(np.max(np.abs(d))), "ncc": ncc(a, b)}


def transform_parameters(matrix) -> dict:
    matrix = np.asarray(matrix, np.float64)
    scale = float(np.linalg.norm(matrix[:2, 0]))
    return {"rotation_degrees_image_xy": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
            "scale_model_px_per_source_px": scale, "source_fov_width_px": 128/scale,
            "crop_center_source_xy": map_points(np.linalg.inv(matrix), [[63.5, 63.5]])[0].tolist()}


def source_marks() -> dict:
    data = json.loads(MANUAL_JSON.read_text(encoding="utf-8"))
    return {item["id"]: [float(item["x"]), float(item["y"])]
            for group in data["landmarks"].values() for item in group}


def read_transform(key: str):
    data = json.loads(TRANSFORMS[key].read_text(encoding="utf-8"))
    item = next(x for x in data["candidates"] if x["name"] == "anchor_tighter")
    return np.asarray(item["original_to_model"], np.float64), data


def transform_report(ma, au, auto_data, marks, detector) -> dict:
    pm, pa = transform_parameters(ma), transform_parameters(au)
    center_shift = np.asarray(pa["crop_center_source_xy"]) - pm["crop_center_source_xy"]
    rows = {}
    for label, xy in marks.items():
        mm = map_points(ma, [xy])[0]
        am = map_points(au, [xy])[0]
        rows[label] = {"manual_source_xy": xy, "manual_model_uv": mm.tolist(),
                       "same_point_auto_model_uv": am.tolist(),
                       "same_point_shift_model_px": float(np.linalg.norm(am-mm)),
                       "model_to_source_round_trip_error_px": float(np.linalg.norm(
                           map_points(np.linalg.inv(au), [am])[0]-xy))}
    error_rows = detector["strategies"]["full_stretch"]["errors_vs_manual"]
    fit_candidate = next(x for x in auto_data["candidates"] if x["name"] == "anchor_tighter")
    fit_labels = [x["name"] for x in fit_candidate["landmarks"] if x["role"] == "fit"]
    corresponding = {}
    for label, item in error_rows.items():
        if label not in marks:
            continue
        source_manual = np.asarray(marks[label])
        source_auto = np.asarray(item["predicted_source_xy"])
        uv_manual = map_points(ma, [source_manual])[0]
        uv_auto = map_points(au, [source_auto])[0]
        corresponding[label] = {
            "pyceph_error_original_px": float(np.linalg.norm(source_auto-source_manual)),
            "manual_model_uv": uv_manual.tolist(), "automatic_model_uv": uv_auto.tolist(),
            "model_displacement_px": float(np.linalg.norm(uv_auto-uv_manual)),
            "manual_original_xy": source_manual.tolist(), "automatic_original_xy": source_auto.tolist(),
        }
    auto_config = json.loads((TRANSFORMS["automatic"].parent / "generated_config.json").read_text())
    fit_items = auto_config["landmarks"]
    target = np.asarray([item["target"] for item in fit_items], np.float64)
    auto_source = np.asarray([item["source"] for item in fit_items], np.float64)
    from ceph_canonicalization import scale_about_model_center
    effects = {}
    for index, item in enumerate(fit_items):
        replaced = auto_source.copy()
        replaced[index] = marks[item["name"]]
        trial = scale_about_model_center(fit_similarity(replaced, target), 1.15)
        tp = transform_parameters(trial)
        effects[item["name"]] = {
            "pyceph_error_source_px": error_rows[item["name"]]["error_px"],
            "crop_center_change_if_replaced_source_px": float(np.linalg.norm(
                np.asarray(tp["crop_center_source_xy"])-pa["crop_center_source_xy"])),
            "rotation_change_if_replaced_degrees": tp["rotation_degrees_image_xy"]-pa["rotation_degrees_image_xy"],
            "fov_change_if_replaced_source_px": tp["source_fov_width_px"]-pa["source_fov_width_px"],
        }
    return {"manual": pm, "automatic": pa,
            "automatic_minus_manual_rotation_degrees": pa["rotation_degrees_image_xy"]-pm["rotation_degrees_image_xy"],
            "automatic_div_manual_scale": pa["scale_model_px_per_source_px"]/pm["scale_model_px_per_source_px"],
            "automatic_minus_manual_fov_source_px": pa["source_fov_width_px"]-pm["source_fov_width_px"],
            "crop_center_delta_source_xy": center_shift.tolist(),
            "crop_center_displacement_source_px": float(np.linalg.norm(center_shift)),
            "manual_to_auto_matrix": (au @ np.linalg.inv(ma)).tolist(),
            "all_curated_landmark_same_point_model_shifts": rows,
            "corresponding_manual_vs_pyceph_marks": corresponding,
            "fit_labels": fit_labels,
            "fit_label_replacement_diagnostic": effects,
            "largest_detector_errors_not_used_in_fit": sorted(
                [(name, item["error_px"]) for name, item in error_rows.items() if name not in fit_labels],
                key=lambda x: -x[1])[:5]}


def optional_ali_comparison(ma, au) -> dict | None:
    paths = {key: BASE / f"ali_{key}" / "probe.json" for key in ("manual", "automatic")}
    if not all(path.is_file() for path in paths.values()):
        return None
    probes = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    for field in ("intensity_hypothesis", "ali_source_revision", "checkpoint_sha256"):
        if probes["manual"].get(field) != probes["automatic"].get(field):
            raise ValueError(f"ALI probe settings differ between runs: {field}")
    names = set(probes["manual"]["model_xyz_continuous_index"]) & set(
        probes["automatic"]["model_xyz_continuous_index"])
    rows = {}
    for name in sorted(names):
        xyz_m = np.asarray(probes["manual"]["model_xyz_continuous_index"][name], np.float64)
        xyz_a = np.asarray(probes["automatic"]["model_xyz_continuous_index"][name], np.float64)
        uv_m = np.asarray(project_drr_xyz(xyz_m), np.float64)
        uv_a = np.asarray(project_drr_xyz(xyz_a), np.float64)
        xy_m = map_points(np.linalg.inv(ma), [uv_m])[0]
        xy_a = map_points(np.linalg.inv(au), [uv_a])[0]
        rows[name] = {"manual_xyz_model_index": xyz_m.tolist(), "automatic_xyz_model_index": xyz_a.tolist(),
                      "xyz_displacement_model_indices": float(np.linalg.norm(xyz_a-xyz_m)),
                      "manual_projected_model_uv": uv_m.tolist(),
                      "automatic_projected_model_uv": uv_a.tolist(),
                      "projected_displacement_model_px": float(np.linalg.norm(uv_a-uv_m)),
                      "manual_projected_original_xy": xy_m.tolist(),
                      "automatic_projected_original_xy": xy_a.tolist(),
                      "projected_displacement_original_px": float(np.linalg.norm(xy_a-xy_m))}
    return {"status": "hypothetical_intensity_ALI_probe_not_anatomical_ground_truth",
            "labels": rows, "count": len(rows),
            "mean_xyz_displacement_model_indices": float(np.mean(
                [x["xyz_displacement_model_indices"] for x in rows.values()])),
            "mean_projected_displacement_model_px": float(np.mean(
                [x["projected_displacement_model_px"] for x in rows.values()])),
            "mean_projected_displacement_original_px": float(np.mean(
                [x["projected_displacement_original_px"] for x in rows.values()]))}


def validate_controlled_configs(manual_dir: Path, automatic_dir: Path) -> None:
    manual = json.loads((manual_dir / "generated_config.json").read_text(encoding="utf-8"))
    automatic = json.loads((automatic_dir / "generated_config.json").read_text(encoding="utf-8"))
    for field in ("source", "target", "orientation", "intensity", "artifact_masks", "candidates"):
        if manual[field] != automatic[field]:
            raise ValueError(f"Canonicalization configs differ beyond input landmarks: {field}")
    for field in ("manual_template_sha256", "template_transform_sha256", "fit_labels"):
        if manual["provenance"][field] != automatic["provenance"][field]:
            raise ValueError(f"Canonicalization template differs: {field}")
    fixed = [(item["name"], item["role"], item["target"]) for item in manual["landmarks"]]
    other = [(item["name"], item["role"], item["target"]) for item in automatic["landmarks"]]
    if fixed != other:
        raise ValueError("Fit landmark labels or template targets differ")


def collect(output) -> dict:
    names = {"coarse_drr": 0, "refined_drr": 1, "coarse_df": 2, "refined_df": 3,
             "pca": 4, "refined_volume": 6, "mandible": 11}
    return {key: output[index].detach().cpu().numpy().squeeze().copy()
            for key, index in names.items()}


def save_visuals(a, b, report):
    BASE.mkdir(parents=True, exist_ok=True)
    # Side-by-side inputs and absolute volume-difference projection.
    with Image.open(INPUTS["manual"]) as x, Image.open(INPUTS["automatic"]) as y:
        canvas = Image.new("RGB", (768, 280), (15, 15, 15))
        for idx, (title, image) in enumerate((("manual", x), ("automatic", y))):
            canvas.paste(image.convert("RGB").resize((256, 256), Image.Resampling.NEAREST), (idx*256, 24))
        diff = np.abs(np.asarray(x, np.int16)-np.asarray(y, np.int16)).astype(np.uint8)
        canvas.paste(Image.fromarray(diff).convert("RGB").resize((256, 256), Image.Resampling.NEAREST), (512, 24))
        draw = ImageDraw.Draw(canvas)
        for idx, title in enumerate(("manual", "automatic", "absolute difference")):
            draw.text((idx*256+6, 5), title, fill="white")
        canvas.save(BASE / "input_comparison.png")
    manual_volume = a["refined_volume"]
    delta = np.max(np.abs(manual_volume-b["refined_volume"]), axis=0)
    gray = np.clip(np.mean(manual_volume, axis=0)*255, 0, 255).astype(np.uint8)
    heat = np.clip(delta / max(float(np.quantile(delta, .99)), 1e-8)*255, 0, 255).astype(np.uint8)
    Image.fromarray(np.stack((np.maximum(gray,heat), gray//2, gray//2), axis=-1)).resize(
        (512, 512), Image.Resampling.NEAREST).save(BASE / "volume_difference_projection.png")
    if report["ali"] is not None:
        source = Path(report["source_path"])
        with Image.open(source) as original:
            overlay = original.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for name, item in report["ali"]["labels"].items():
            mx, my = item["manual_projected_original_xy"]
            ax, ay = item["automatic_projected_original_xy"]
            draw.line((mx, my, ax, ay), fill=(255, 255, 255), width=4)
            draw.ellipse((mx-11, my-11, mx+11, my+11), outline=(0, 240, 255), width=5)
            draw.ellipse((ax-11, ay-11, ax+11, ay+11), outline=(255, 220, 0), width=5)
            draw.text((ax+14, ay-12), name, fill=(255, 220, 0))
        overlay.save(BASE / "ali_projection_original_overlay.png")


def main():
    import torch
    import SimpleITK as sitk
    started = time.perf_counter()
    manual_hash_before = __import__("hashlib").sha256(MANUAL_JSON.read_bytes()).hexdigest()
    for image in INPUTS.values():
        validate_inputs(ROOT / "models", image, 60)
    ma, mdata = read_transform("manual")
    au, adata = read_transform("automatic")
    validate_controlled_configs(TRANSFORMS["manual"].parent, TRANSFORMS["automatic"].parent)
    if mdata["source"]["sha256"] != adata["source"]["sha256"]:
        raise ValueError("Inputs do not originate from the same ceph")
    detector = json.loads(DETECTOR_REPORT.read_text(encoding="utf-8"))
    if detector["source"]["sha256"] != mdata["source"]["sha256"]:
        raise ValueError("Detector evaluation originated from a different ceph")
    if detector["manual_reference_sha256"] != manual_hash_before:
        raise ValueError("Detector evaluation used a different manual landmark export")
    marks = source_marks()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_class()(60, str(ROOT / "models")).to(device)
    checkpoint = torch.load(ROOT / "models/cbct_c2f_model_ckpt.tar", map_location=device,
                            weights_only=True, mmap=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    outputs = {}
    for key, path in INPUTS.items():
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255
        tensor = torch.from_numpy(image).reshape(1, 1, 128, 128).to(device)
        with torch.inference_mode():
            result = model(tensor)
        outputs[key] = collect(result)
        del result, tensor
        # Confirm this analysis pass matches the exact project demo runner.
        saved = sitk.GetArrayFromImage(sitk.ReadImage(str(BASE / key / "anchor_tighter_128_refine.nii.gz")))
        if not np.allclose(outputs[key]["refined_volume"], saved, rtol=0, atol=1e-6):
            raise RuntimeError(f"Analysis forward disagrees with published runner output: {key}")
        saved_seg = sitk.GetArrayFromImage(sitk.ReadImage(str(BASE / key / "anchor_tighter_128_seg_reg.nii.gz")))
        if not np.array_equal(outputs[key]["mandible"] >= .5, saved_seg > 0):
            raise RuntimeError(f"Analysis segmentation disagrees with published runner output: {key}")
    a, b = outputs["manual"], outputs["automatic"]
    image_a = np.asarray(Image.open(INPUTS["manual"]).convert("L"), np.float32)/255
    image_b = np.asarray(Image.open(INPUTS["automatic"]).convert("L"), np.float32)/255
    comparisons = {name: pair_metrics(a[name], b[name])
                   for name in ("coarse_drr", "refined_drr", "coarse_df", "refined_df", "pca", "refined_volume")}
    for name in ("coarse_df", "refined_df"):
        diff = b[name]-a[name]
        mag = np.linalg.norm(diff, axis=-1)
        comparisons[name]["vector_delta_mean_model_voxels"] = float(mag.mean())
        comparisons[name]["vector_delta_p95_model_voxels"] = float(np.quantile(mag, .95))
        comparisons[name]["vector_delta_max_model_voxels"] = float(mag.max())
    seg_a, seg_b = a["mandible"] >= .5, b["mandible"] >= .5
    centroid_a = np.argwhere(seg_a).mean(axis=0)
    centroid_b = np.argwhere(seg_b).mean(axis=0)
    segmentation = {"manual_voxels": int(seg_a.sum()), "automatic_voxels": int(seg_b.sum()),
                    "dice": float(2*np.count_nonzero(seg_a & seg_b)/(seg_a.sum()+seg_b.sum())),
                    "manual_centroid_zyx": centroid_a.tolist(), "automatic_centroid_zyx": centroid_b.tolist(),
                    "centroid_delta_zyx_voxels": (centroid_b-centroid_a).tolist(),
                    "centroid_displacement_grid_voxels": float(np.linalg.norm(centroid_b-centroid_a))}
    report = {"status": "manual_as_experimental_reference_not_3d_ground_truth",
              "device": str(device), "torch": torch.__version__,
              "source_sha256": mdata["source"]["sha256"], "source_path": mdata["source"]["path"],
              "manual_landmarks_sha256": manual_hash_before,
              "input_metrics": pair_metrics(image_a, image_b),
              "transform": transform_report(ma, au, adata, marks, detector),
              "ali": optional_ali_comparison(ma, au),
              "outputs": comparisons, "segmentation": segmentation,
              "pca_values_manual": a["pca"].tolist(), "pca_values_automatic": b["pca"].tolist(),
              "runtime_seconds": time.perf_counter()-started}
    report["outputs"]["pca"]["l2_delta"] = float(np.linalg.norm(b["pca"]-a["pca"]))
    report["outputs"]["pca"]["manual_l2"] = float(np.linalg.norm(a["pca"]))
    report["outputs"]["pca"]["automatic_l2"] = float(np.linalg.norm(b["pca"]))
    save_visuals(a, b, report)
    if __import__("hashlib").sha256(MANUAL_JSON.read_bytes()).hexdigest() != manual_hash_before:
        raise RuntimeError("Curated manual landmark file changed")
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "comparison.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(BASE / "comparison.json")


if __name__ == "__main__":
    main()
