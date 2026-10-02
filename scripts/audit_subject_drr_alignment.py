"""Compare input cephs with their own SC-DREG DRRs; diagnostic only."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sc_dreg_compat import ROOT
from study_sc_dreg_input_sensitivity import edge_magnitude, ncc


CASES = {
    "previous_zoomed_anchor_tighter": ROOT / "outputs/subject_inspection/inference/manual_masked_anchor_tighter_recreated_128",
    "recommended_masked_tighter_fov": ROOT / "outputs/subject_inspection/recommended_inference/manual_masked_tighter_fov_recreated_128",
    "published_04002": ROOT / "outputs/sc_dreg_demo/04002",
}


def metrics(raw: np.ndarray, drr: np.ndarray, torch) -> dict:
    from pytorch_msssim import ssim

    ea, eb = edge_magnitude(raw), edge_magnitude(drr)
    ma = ea > max(float(np.quantile(ea, .85)), 1e-8)
    mb = eb > max(float(np.quantile(eb, .85)), 1e-8)
    return {"ncc": ncc(raw, drr),
            "ssim": float(ssim(torch.from_numpy(raw.copy())[None, None],
                               torch.from_numpy(drr.copy())[None, None], data_range=1)),
            "edge_ncc": ncc(ea, eb),
            "top15_edge_dice": float(2 * np.count_nonzero(ma & mb) / (ma.sum() + mb.sum()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/subject_drr_audit")
    args = parser.parse_args()
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import SimpleITK as sitk

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {name: stem for name, stem in CASES.items()
             if Path(str(stem) + "_raw.png").is_file() and Path(str(stem) + "_refine_drr.png").is_file()}
    if "recommended_masked_tighter_fov" not in cases:
        raise FileNotFoundError("Run the recommended masked tighter_fov inference first")
    fig, axes = plt.subplots(len(cases), 3, figsize=(10, 3.4 * len(cases)),
                             constrained_layout=True, squeeze=False)
    report = {"meaning": "input-to-own-DRR self-consistency, not 3D anatomical accuracy", "cases": {}}
    for row, (name, stem) in enumerate(cases.items()):
        raw = np.asarray(Image.open(str(stem) + "_raw.png"), np.float32) / 255
        drr = np.asarray(Image.open(str(stem) + "_refine_drr.png"), np.float32) / 255
        result = metrics(raw, drr, torch)
        result["orientation_ncc"] = {
            "as_saved": ncc(raw, drr), "flip_x": ncc(raw, drr[:, ::-1]),
            "flip_y": ncc(raw, drr[::-1, :]), "rotate_180": ncc(raw, drr[::-1, ::-1])}
        report["cases"][name] = result
        overlay = np.stack((raw, drr, np.zeros_like(raw)), axis=-1)
        for column, (image, title) in enumerate(((raw, "input"), (drr, "refined DRR"),
                                                  (overlay, "red=input, green=DRR"))):
            axis = axes[row, column]
            axis.imshow(image, cmap="gray" if column < 2 else None, vmin=0, vmax=1)
            axis.set_title(name + "\n" + title if column == 0 else title, fontsize=9)
            axis.axis("off")
    if "previous_zoomed_anchor_tighter" in cases:
        a = sitk.GetArrayFromImage(sitk.ReadImage(str(cases["previous_zoomed_anchor_tighter"]) + "_refine.nii.gz"))
        b = sitk.GetArrayFromImage(sitk.ReadImage(str(cases["recommended_masked_tighter_fov"]) + "_refine.nii.gz"))
        sa = sitk.GetArrayFromImage(sitk.ReadImage(str(cases["previous_zoomed_anchor_tighter"]) + "_seg_reg.nii.gz")) >= .5
        sb = sitk.GetArrayFromImage(sitk.ReadImage(str(cases["recommended_masked_tighter_fov"]) + "_seg_reg.nii.gz")) >= .5
        report["zoomed_vs_recommended_outputs"] = {
            "refined_volume_mae": float(np.mean(np.abs(a - b))),
            "refined_volume_correlation": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
            "registered_mandible_dice": float(2 * np.count_nonzero(sa & sb) / (sa.sum() + sb.sum()))}
    if "published_04002" in cases:
        subject = cases["recommended_masked_tighter_fov"]
        published = cases["published_04002"]
        def png(name: str, suffix: str) -> np.ndarray:
            return np.asarray(Image.open(str(cases[name]) + suffix), np.float32) / 255
        input_subject = png("recommended_masked_tighter_fov", "_raw.png")
        input_published = png("published_04002", "_raw.png")
        drr_subject = png("recommended_masked_tighter_fov", "_refine_drr.png")
        drr_published = png("published_04002", "_refine_drr.png")
        volume_subject = sitk.GetArrayFromImage(sitk.ReadImage(str(subject) + "_refine.nii.gz"))
        volume_published = sitk.GetArrayFromImage(sitk.ReadImage(str(published) + "_refine.nii.gz"))
        jaw_subject = sitk.GetArrayFromImage(sitk.ReadImage(str(subject) + "_seg_reg.nii.gz")) >= .5
        jaw_published = sitk.GetArrayFromImage(sitk.ReadImage(str(published) + "_seg_reg.nii.gz")) >= .5
        report["subject_vs_published_04002"] = {
            "input_ncc": ncc(input_subject, input_published),
            "refined_drr_ncc": ncc(drr_subject, drr_published),
            "refined_drr_mae_unit_range": float(np.mean(np.abs(drr_subject - drr_published))),
            "refined_volume_ncc": ncc(volume_subject, volume_published),
            "registered_mandible_dice": float(2 * np.count_nonzero(jaw_subject & jaw_published)
                                               / (jaw_subject.sum() + jaw_published.sum())),
            "interpretation_limit": "Similar projections do not prove identical fields or that the model ignores the input"}
    fig.savefig(args.output_dir / "input_drr_comparison.png", dpi=150)
    plt.close(fig)
    (args.output_dir / "alignment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / "alignment.json")


if __name__ == "__main__":
    main()
