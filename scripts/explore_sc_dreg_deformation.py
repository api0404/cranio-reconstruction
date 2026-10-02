"""Summarize captured SC-DREG fields and all 60 coarse PCA directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sc_dreg_compat import ROOT


SHAPE = (128, 128, 128, 3)


def magnitude_summary(field: np.ndarray, mask: np.ndarray | None = None) -> dict:
    values = field[mask] if mask is not None else field.reshape(-1, 3)
    length = np.linalg.norm(values.astype(np.float64), axis=-1)
    return {"mean": float(length.mean()), "median": float(np.median(length)),
            "p95": float(np.quantile(length, .95)), "p99": float(np.quantile(length, .99)),
            "max": float(length.max()), "rms": float(np.sqrt(np.mean(length**2)))}


def save_map(path: Path, magnitude: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    views = ((magnitude.max(axis=0), "max over z; x right, y down"),
             (magnitude.max(axis=1), "max over y; x right, z down"),
             (magnitude.max(axis=2), "max over x; y right, z down"))
    vmax = max(float(np.quantile(magnitude, .99)), 1e-8)
    for axis, (view, label) in zip(axes, views):
        image = axis.imshow(view, origin="upper", cmap="magma", vmin=0, vmax=vmax)
        axis.set_title(label, fontsize=9)
    fig.colorbar(image, ax=axes, label="model grid index units")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_vectors(path: Path, field: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fixed planes make cases/modes comparable, with actual x/y component arrows.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for axis, z in zip(axes, (32, 64, 96)):
        yy, xx = np.mgrid[8:128:8, 8:128:8]
        vectors = field[z, yy, xx]
        color = np.linalg.norm(vectors, axis=-1)
        axis.quiver(xx, yy, vectors[..., 0], vectors[..., 1], color,
                   cmap="viridis", angles="xy", scale_units="xy", scale=1, width=.004)
        axis.set(xlim=(0, 127), ylim=(127, 0), aspect="equal", title=f"z={z}; in-plane x/y vectors")
    fig.suptitle(title + " (arrows show actual displacement, no display scaling)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_vectors_3d(path: Path, field: np.ndarray, reference: np.ndarray, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zz, yy, xx = np.mgrid[8:128:16, 8:128:16, 8:128:16]
    keep = reference[zz, yy, xx] >= .35
    vectors = field[zz, yy, xx][keep]
    coords = (xx[keep], yy[keep], zz[keep])
    lengths = np.linalg.norm(vectors, axis=-1)
    fig = plt.figure(figsize=(9, 8))
    axis = fig.add_subplot(111, projection="3d")
    colors = plt.get_cmap("viridis")(lengths / max(np.quantile(lengths, .95), 1e-8))
    axis.quiver(*coords, vectors[:, 0], vectors[:, 1], vectors[:, 2],
                colors=colors, length=1, normalize=False, arrow_length_ratio=.25)
    axis.set(xlim=(0, 127), ylim=(0, 127), zlim=(0, 127), xlabel="x", ylabel="y",
             zlabel="z", title=title + "\nreference intensity >=0.35; stride 16")
    axis.set_box_aspect((1, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def capture(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        pca = data["pca"].copy()
        coarse = data["coarse"].copy()
        refined = data["refined"].copy()
    if pca.shape != (60,) or coarse.shape != SHAPE or refined.shape != SHAPE:
        raise ValueError(f"Invalid deformation capture: {path}")
    if not all(np.isfinite(item).all() for item in (pca, coarse, refined)):
        raise ValueError(f"Nonfinite deformation capture: {path}")
    return pca, coarse, refined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path, default=ROOT / "outputs/sc_dreg_demo/04002_deformation.npz")
    parser.add_argument("--subject", type=Path, required=True,
                        help="Capture from the selected current-subject 128x128 input")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/deformation_exploration")
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()
    import SimpleITK as sitk

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mandible = sitk.GetArrayFromImage(sitk.ReadImage(str(args.models / "seg_mandible.nii.gz"))) >= .5
    reference = sitk.GetArrayFromImage(sitk.ReadImage(str(args.models / "ref.nii.gz")))
    if mandible.shape != SHAPE[:3]:
        raise ValueError("Mandible reference mask shape mismatch")
    mean = np.load(args.models / "mean4.npy", mmap_mode="r", allow_pickle=False)
    basis = np.load(args.models / "coeff4.npy", mmap_mode="r", allow_pickle=False)
    if mean.shape != (np.prod(SHAPE),) or basis.shape[0] < 60 or basis.shape[1] != np.prod(SHAPE):
        raise ValueError("Unexpected PCA asset shapes")
    cases = {}
    captures = {}
    for label, path in (("04002", args.published), ("subject", args.subject)):
        pca, coarse, refined = capture(path)
        captures[label] = pca
        delta = refined - coarse  # Difference of complete fields, not added by upstream.
        case_dir = args.output_dir / label
        case_dir.mkdir(exist_ok=True)
        case = {"capture": str(path), "pca": pca.tolist(),
                "coarse": magnitude_summary(coarse), "refined": magnitude_summary(refined),
                "refined_minus_coarse": magnitude_summary(delta),
                "mandible_region": {"coarse": magnitude_summary(coarse, mandible),
                                    "refined": magnitude_summary(refined, mandible),
                                    "refined_minus_coarse": magnitude_summary(delta, mandible)}}
        for name, field in (("coarse", coarse), ("refined", refined), ("refinement_change", delta)):
            magnitude = np.linalg.norm(field, axis=-1)
            index = np.unravel_index(np.argmax(magnitude), magnitude.shape)
            case[name + "_maximum_zyx"] = list(map(int, index))
            top = np.argwhere(magnitude >= np.quantile(magnitude, .99))
            case[name + "_top1pct_bbox_zyx"] = [top.min(axis=0).tolist(), top.max(axis=0).tolist()]
            save_map(case_dir / f"{name}_magnitude.png", magnitude, f"{label}: {name}")
        save_vectors(case_dir / "coarse_vectors.png", coarse, f"{label}: coarse")
        save_vectors(case_dir / "refined_vectors.png", refined, f"{label}: refined")
        save_vectors_3d(case_dir / "coarse_vectors_3d.png", coarse, reference, f"{label}: coarse field")
        save_vectors_3d(case_dir / "refined_vectors_3d.png", refined, reference, f"{label}: refined field")
        # Reconstruct with the same float32 arithmetic and ordering as upstream.
        # Small CPU BLAS differences are reported, never silently accepted.
        reconstructed = (pca.astype(np.float32) @ basis[:60].astype(np.float32)
                         + mean.astype(np.float32)).reshape(SHAPE)
        case["coarse_reconstruction_max_abs_error"] = float(np.max(np.abs(reconstructed - coarse)))
        cases[label] = case
    modes = []
    for i in range(60):
        # One row only: no full 6 GiB basis copy or 60-row mode stack.
        field = np.asarray(basis[i], dtype=np.float32).reshape(SHAPE)
        global_stats = magnitude_summary(field)
        mandible_stats = magnitude_summary(field, mandible)
        norm = np.linalg.norm(field, axis=-1)
        maximum = np.unravel_index(np.argmax(norm), norm.shape)
        high = np.argwhere(norm >= np.quantile(norm, .99))
        modes.append({"mode_index_zero_based": i, "unit_coefficient": global_stats,
                      "mandible_reference_mask": mandible_stats,
                      "unit_maximum_zyx": list(map(int, maximum)),
                      "unit_top1pct_bbox_zyx": [high.min(axis=0).tolist(), high.max(axis=0).tolist()],
                      "contributions": {case: {"coefficient": float(pca[i]),
                                              "global_rms": abs(float(pca[i])) * global_stats["rms"],
                                              "mandible_rms": abs(float(pca[i])) * mandible_stats["rms"]}
                                        for case, pca in captures.items()}})
    ranking = {case: sorted(range(60), key=lambda i: modes[i]["contributions"][case]["global_rms"],
                           reverse=True) for case in captures}
    for case, ordered in ranking.items():
        cases[case]["mode_ranking_global_rms_zero_based"] = ordered
    selected = sorted(set(ranking["04002"][:args.top] + ranking["subject"][:args.top]))
    mode_dir = args.output_dir / "modes"
    mode_dir.mkdir(exist_ok=True)
    for i in selected:
        field = np.asarray(basis[i], dtype=np.float32).reshape(SHAPE)
        mag = np.linalg.norm(field, axis=-1)
        save_map(mode_dir / f"mode_{i:02d}_unit_magnitude.png", mag, f"Mode {i}: unit coefficient")
        save_vectors(mode_dir / f"mode_{i:02d}_unit_vectors.png", field, f"Mode {i}: unit coefficient")
        save_vectors_3d(mode_dir / f"mode_{i:02d}_unit_vectors_3d.png", field, reference,
                        f"Mode {i}: unit coefficient")
    report = {"coordinates": "fields[z,y,x,(dx,dy,dz)] and model-grid index units, not mm",
              "model": "coarse = para @ float32(coeff4[:60]) + float32(mean4); refined is a replacement field",
              "ranking_metric": "abs(predicted coefficient) times basis global RMS magnitude; individual linear term, not orthogonal variance share",
              "basis_rows": 60, "modes": modes, "cases": cases, "visualized_modes": selected}
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir / "analysis.json")


if __name__ == "__main__":
    main()
