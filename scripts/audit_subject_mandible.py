"""Compare curated 2D jaw marks with the projected SC-DREG mandible mask.

The mask is the model's warped reference mandible, not a subject CT label.
Distance to its silhouette is a proxy and does not identify model Gonion.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from ceph_canonicalization import map_points
from sc_dreg_compat import ROOT, UPSTREAM_SRC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/subject_inspection/recommended_input")
    parser.add_argument("--inference-dir", type=Path, default=ROOT / "outputs/subject_inspection/recommended_inference")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/subject_drr_audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import SimpleITK as sitk
    import torch
    from scipy.ndimage import binary_erosion
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(UPSTREAM_SRC))
    from drr import GenerateSeg

    stem = "manual_masked_tighter_fov_recreated_128"
    raw = np.asarray(Image.open(args.inference_dir / f"{stem}_raw.png"))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(args.inference_dir / f"{stem}_seg_reg.nii.gz")))
    with torch.inference_mode():
        projected = GenerateSeg()(torch.from_numpy(seg.astype(np.float32))[None, None]).squeeze().numpy()
    mask = projected >= .5
    boundary = mask & ~binary_erosion(mask)
    yy, xx = np.nonzero(boundary)
    contour_xy = np.stack((xx, yy), axis=1)
    transform = json.loads((args.input_dir / "transforms.json").read_text())
    matrix = np.asarray(transform["candidates"][0]["original_to_model"], dtype=np.float64)
    marks_json = json.loads((ROOT / "data/private/high_value_ceph_landmarks.json").read_text())
    marks = {item["id"]: [item["x"], item["y"]]
             for group in marks_json["landmarks"].values() for item in group}
    labels = ("Go", "Go_constructed", "Me", "Pog", "B")
    mapped = {label: map_points(matrix, [marks[label]])[0] for label in labels}
    records = {}
    for label, point in mapped.items():
        distances = np.linalg.norm(contour_xy - point, axis=1)
        nearest = contour_xy[np.argmin(distances)]
        records[label] = {"curated_model_xy": point.tolist(),
                          "nearest_projected_mask_boundary_xy": nearest.tolist(),
                          "distance_model_pixels": float(distances.min()),
                          "note": "Nearest mask edge is not a model anatomical landmark"}
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(raw, cmap="gray", vmin=0, vmax=255)
    ax.contour(mask.astype(float), levels=[.5], colors=["cyan"], linewidths=1.4)
    for label, item in records.items():
        x, y = item["curated_model_xy"]
        ax.plot(x, y, "o", color="yellow", markersize=4)
        ax.text(x + 1, y - 1, label, color="yellow", fontsize=9)
    go, me = mapped["Go"], mapped["Me"]
    ax.plot([go[0], me[0]], [go[1], me[1]], color="orange", linewidth=1.5)
    ax.set(xlim=(0, 127), ylim=(127, 0), title="Curated jaw marks vs projected warped-reference mandible\n"
           "cyan=model mask; yellow=manual marks; orange=manual Go-Me line")
    fig.savefig(args.output_dir / "mandible_mark_overlay.png", dpi=180)
    plt.close(fig)
    report = {"meaning": "2D comparison of curated marks with model-warped reference mask silhouette",
              "model_gonion_or_fma_estimated": False,
              "projection_source": "upstream GenerateSeg applied to saved rounded refined mandible mask",
              "boundary_pixels": int(len(contour_xy)), "marks": records}
    (args.output_dir / "mandible_alignment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / "mandible_alignment.json")


if __name__ == "__main__":
    main()
