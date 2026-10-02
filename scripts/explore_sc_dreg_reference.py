"""Export model-grid reference meshes and a measured intensity/geometry report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sc_dreg_compat import ROOT


THRESHOLDS = (0.35, 0.55, 0.75)  # Deliberate visual sweep, not HU cutoffs.


def stats(a: np.ndarray) -> dict:
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "std": float(a.std()), "quantiles": {str(q): float(np.quantile(a, q))
                                             for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)}}


def write_binary_ply(path: Path, vertices_xyz: np.ndarray, faces: np.ndarray) -> None:
    """Write triangles in model (x,y,z) index coordinates, no implicit spacing."""
    vertices = np.asarray(vertices_xyz, dtype="<f4")
    triangles = np.asarray(faces, dtype="<i4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment coordinates: SC-DREG model grid index (x,y,z); units are not mm\n"
              f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
              f"element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n")
    face_records = np.empty(len(triangles), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    face_records["count"] = 3
    face_records["indices"] = triangles
    with path.open("wb") as output:
        output.write(header.encode("ascii"))
        vertices.tofile(output)
        face_records.tofile(output)


def mesh_at_level(volume: np.ndarray, level: float, path: Path) -> dict:
    from skimage.measure import marching_cubes

    # Volume is [z,y,x]; marching_cubes returns vertices in the same order.
    zyx, faces, _, _ = marching_cubes(volume, level=level, spacing=(1, 1, 1))
    xyz = zyx[:, ::-1].copy()
    write_binary_ply(path, xyz, faces)
    return {"file": path.name, "vertices": len(xyz), "triangles": len(faces),
            "bounds_xyz": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reference_exploration")
    args = parser.parse_args()
    import SimpleITK as sitk

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"coordinates": "vertices=(x,y,z) model array indices; volumes=(z,y,x); not mm",
              "thresholds": {}, "assets": {}}
    for name in ("ref", "seg_mandible"):
        path = args.models / f"{name}.nii.gz"
        image = sitk.ReadImage(str(path))
        volume = sitk.GetArrayFromImage(image)
        if volume.shape != (128, 128, 128) or not np.isfinite(volume).all():
            raise ValueError(f"Unexpected {name} volume")
        report["assets"][name] = {
            "shape_zyx": list(volume.shape), "dtype": str(volume.dtype),
            "spacing_header": image.GetSpacing(), "origin_header": image.GetOrigin(),
            "direction_header": image.GetDirection(), "intensity": stats(volume),
            "nonzero_voxels": int(np.count_nonzero(volume)),
            "histogram_counts_0_to_1_20_bins": np.histogram(volume, bins=np.linspace(0, 1, 21))[0].tolist()}
        if name == "ref":
            reference = volume
        else:
            mandible = volume
    mask = mandible >= .5
    report["assets"]["seg_mandible"]["mask_voxels_at_0.5"] = int(mask.sum())
    report["assets"]["seg_mandible"]["mask_bounds_zyx"] = [np.argwhere(mask).min(axis=0).tolist(),
                                                            np.argwhere(mask).max(axis=0).tolist()]
    report["mandible_mesh"] = mesh_at_level(mandible, .5, args.output_dir / "reference_mandible.ply")
    for threshold in THRESHOLDS:
        tag = str(threshold).replace(".", "p")
        report["thresholds"][str(threshold)] = {
            "voxels_at_or_above": int(np.count_nonzero(reference >= threshold)),
            "mesh": mesh_at_level(reference, threshold, args.output_dir / f"reference_intensity_{tag}.ply")}
    (args.output_dir / "reference.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / "reference.json")


if __name__ == "__main__":
    main()
