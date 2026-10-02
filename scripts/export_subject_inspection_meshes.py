"""Export visual SC-DREG subject surfaces and multi-view quick previews."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from explore_sc_dreg_reference import stats, write_binary_ply
from sc_dreg_compat import ROOT


THRESHOLDS = (0.45, 0.55, 0.65)


def mesh(volume: np.ndarray, level: float, path: Path) -> tuple[np.ndarray, dict]:
    from skimage.measure import marching_cubes

    zyx, faces, _, _ = marching_cubes(volume, level=level)
    xyz = zyx[:, ::-1].copy()  # PLY coordinates are x,y,z model indices.
    write_binary_ply(path, xyz, faces)
    return xyz, {"file": path.name, "level": level, "vertices": len(xyz),
                 "triangles": len(faces), "bounds_xyz": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()]}


def preview(path: Path, skull: np.ndarray, mandible: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 5.4), constrained_layout=True)
    views = ((20, -90, "view along y"), (10, 0, "view along x"), (25, -45, "oblique grid view"))
    for number, (elevation, azimuth, label) in enumerate(views, 1):
        axis = fig.add_subplot(1, 3, number, projection="3d")
        points = skull[::max(1, len(skull) // 45000)]
        jaw = mandible[::max(1, len(mandible) // 9000)]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=.18,
                     c="#665d53", alpha=.34, rasterized=True)
        axis.scatter(jaw[:, 0], jaw[:, 1], jaw[:, 2], s=.25,
                     c="#16b7c7", alpha=.45, rasterized=True)
        axis.set(xlim=(0, 127), ylim=(0, 127), zlim=(0, 127),
                 xlabel="x", ylabel="y", zlabel="z", title=label)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
    fig.suptitle(title + " | gray=intensity isosurface, cyan=warped reference mandible\n"
                 "model grid index units; these are point previews of exported PLY surfaces")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path, default=ROOT / "outputs/subject_inspection/recommended_inference")
    parser.add_argument("--stem", default="manual_masked_tighter_fov_recreated_128")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/subject_inspection/recommended_meshes")
    args = parser.parse_args()
    import SimpleITK as sitk

    volume_path = args.inference_dir / f"{args.stem}_refine.nii.gz"
    segment_path = args.inference_dir / f"{args.stem}_seg_reg.nii.gz"
    volume_image = sitk.ReadImage(str(volume_path))
    segment_image = sitk.ReadImage(str(segment_path))
    volume = sitk.GetArrayFromImage(volume_image)
    segment = sitk.GetArrayFromImage(segment_image)
    if volume.shape != (128, 128, 128) or segment.shape != volume.shape:
        raise ValueError("Expected aligned 128^3 subject output and mandible mask")
    if not np.isfinite(volume).all() or not np.isfinite(segment).all():
        raise ValueError("Nonfinite subject outputs")
    if not np.isin(np.unique(segment), [0, 1]).all():
        raise ValueError("Expected rounded warped mandible mask")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jaw_vertices, jaw = mesh(segment, .5, args.output_dir / "subject_warped_reference_mandible.ply")
    sweep = {}
    for level in THRESHOLDS:
        tag = str(level).replace(".", "p")
        vertices, record = mesh(volume, level, args.output_dir / f"subject_intensity_{tag}.ply")
        record["voxels_at_or_above"] = int(np.count_nonzero(volume >= level))
        sweep[str(level)] = record
        preview(args.output_dir / f"subject_intensity_{tag}_views.png", vertices, jaw_vertices,
                f"SC-DREG subject, isovalue {level}")
    report = {
        "input_status": "recommended_masked_tighter_fov_recreation_from_rounded_prior_transform_with_curated_manual_checks",
        "refined_volume": str(volume_path), "registered_mandible_mask": str(segment_path),
        "volume_sha256": hashlib.sha256(volume_path.read_bytes()).hexdigest(),
        "segmentation_sha256": hashlib.sha256(segment_path.read_bytes()).hexdigest(),
        "coordinates": "PLY vertices=(x,y,z) model-grid indices; NIfTI arrays=(z,y,x); not mm",
        "nifti_spacing_header": volume_image.GetSpacing(),
        "nifti_origin_header": volume_image.GetOrigin(),
        "nifti_direction_header": volume_image.GetDirection(),
        "intensity": stats(volume),
        "histogram_counts_0_to_1_20_bins": np.histogram(volume, bins=np.linspace(0, 1, 21))[0].tolist(),
        "threshold_sweep": sweep,
        "recommended_visual_level": .55,
        "recommendation_scope": "visual starting point only; not a validated bone or HU threshold",
        "mandible": {**jaw, "voxels": int(np.count_nonzero(segment)),
                     "provenance": "rounded SC-DREG refined warp of supplied reference mandible mask; not independently segmented subject anatomy"},
        "preview_type": "subsampled mesh-vertex render, not a Blender/Slicer screenshot"}
    (args.output_dir / "extraction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / "extraction.json")


if __name__ == "__main__":
    main()
