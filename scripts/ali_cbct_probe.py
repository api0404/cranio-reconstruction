"""Optional ALI-CBCT compatibility probe for an SC-DREG volume.

This is an experiment with synthetic, uncalibrated geometry. The intensity
mapping is a stated hypothesis and does not create Hounsfield units or mm.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOLUME = ROOT / "outputs/sc_dreg_demo/04002_refine.nii.gz"
DEFAULT_ALI = ROOT / "vendor/slicer-automated-dental-tools/ALI_CBCT/ALI_CBCT.py"
DEFAULT_MODELS = ROOT / "models/ali-cbct/weights"
DEFAULT_PYTHON = ROOT / ".venv-ali/Scripts/python.exe"
ARCHIVE_GROUPS = {"S": "Cranial_Base", "N": "Cranial_Base", "Ba": "Cranial_Base",
                  "A": "Upper_Bones_v2", "ANS": "Upper_Bones_v2", "PNS": "Upper_Bones_v2",
                  "B": "Lower_Bones_2", "Pog": "Lower_Bones_2", "Me": "Lower_Bones_2"}


def native_histogram_cast(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Mirror ALI CorrectHisto's 1st/99th percentile histogram and int16 cast."""
    image = np.asarray(array, dtype=np.float32)
    if not np.isfinite(image).all() or float(image.min()) == float(image.max()):
        raise ValueError("ALI input must have finite, nonconstant intensity")
    histogram = np.histogram(image, 1000)[0]
    cumulative = np.cumsum(histogram)
    cumulative = (cumulative - cumulative.min()) / (cumulative.max() - cumulative.min())
    span = float(image.max() - image.min())
    lo = max(float(np.argmax(cumulative > 0.01) * span / 1000 + image.min()), -1500)
    hi = min(float(np.argmax(cumulative > 0.99) * span / 1000 + image.min()), 4000)
    return np.clip(image, lo, hi).astype(np.int16), lo, hi


def project_drr_xyz(xyz: tuple[float, float, float]) -> tuple[float, float]:
    """SC-DREG DRR sampling locus for a point, in 128-pixel coordinates.

    This uses GenerateDRR's affine grid and 180-degree image rotation. It is
    a geometric projection, not evidence that the NIfTI axes are anatomical.
    """
    x, y, z = xyz
    scale = (101 + (117 - 101) * x / 127) / 128
    return (63.5 - (y - 63.5) / scale, 63.5 - (z - 63.5) / scale)


def correct_ali_export_xyz(position: list[float], fine_origin: tuple[float, float, float]) -> list[float]:
    """Undo ALI SavePredictedLandmarks' abs(origin) offset for identity axes.

    The upstream exporter writes ``index * spacing - abs(origin)`` whereas
    the true physical position is ``index * spacing + origin``. The output
    of this function is still in SC-DREG's uncalibrated coordinate units.
    """
    return (np.asarray(position) + np.asarray(fine_origin) + np.abs(fine_origin)).tolist()


def checkpoint_paths(root: Path, label: str) -> dict[str, Path]:
    matches = {}
    for scale in ("1", "0-3"):
        found = sorted(root.glob(f"*/{label}/{scale}/{label}_Net_{scale}.pth"))
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one {label} checkpoint for scale {scale} under {root}; found {len(found)}")
        matches[scale] = found[0]
    return matches


def ensure_checkpoints(root: Path, archive_dir: Path, labels: list[str]) -> None:
    """Extract only selected pretrained pairs from local release archives."""
    for label in labels:
        if label not in ARCHIVE_GROUPS:
            raise ValueError(f"Unsupported probe label {label}; supported: {sorted(ARCHIVE_GROUPS)}")
        group = ARCHIVE_GROUPS[label]
        archive = archive_dir / f"{group}.zip"
        for scale in ("1", "0-3"):
            destination = root / group / label / scale / f"{label}_Net_{scale}.pth"
            if destination.is_file():
                continue
            if not archive.is_file():
                raise FileNotFoundError(f"Missing ALI weights: {destination}; download {archive.name} into {archive_dir}")
            member = f"{group}/{label}/{scale}/{label}_Net_{scale}.pth"
            with zipfile.ZipFile(archive) as package:
                if member not in package.namelist():
                    raise FileNotFoundError(f"Expected {member} in {archive}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_suffix(".partial")
                with package.open(member) as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target)
                partial.replace(destination)


def read_markups(directory: Path, patient_id: str, after_ns: int = 0) -> dict[str, list[float]]:
    points = {}
    for path in directory.glob(f"{patient_id}_lm_Pred_*.mrk.json"):
        if path.stat().st_mtime_ns < after_ns:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for point in data["markups"][0]["controlPoints"]:
            if data["markups"][0]["coordinateSystem"] != "LPS":
                raise ValueError(f"Unexpected coordinate system in {path}")
            points[point["label"]] = point["position"]
    return points


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def add_coordinate_analysis(report: dict, output_dir: Path, prepared: Path,
                            source_image: sitk.Image) -> None:
    fine_paths = list((output_dir / "temp").glob(f"{prepared.stem}_sp0-3.nii.gz"))
    if len(fine_paths) != 1:
        raise RuntimeError("ALI fine-scale resampled image is missing; inspect ali.log")
    fine = sitk.ReadImage(str(fine_paths[0]))
    identity = (1., 0., 0., 0., 1., 0., 0., 0., 1.)
    if fine.GetDirection() != source_image.GetDirection() or source_image.GetDirection() != identity:
        raise ValueError("This probe's ALI export correction requires identity image direction")
    report["ali_fine_origin_xyz"] = list(fine.GetOrigin())
    report["ali_export_correction_xyz"] = (np.asarray(fine.GetOrigin()) + np.abs(fine.GetOrigin())).tolist()
    report["ali_corrected_physical_xyz_uncalibrated"] = {
        label: correct_ali_export_xyz(point, fine.GetOrigin())
        for label, point in report["ali_markups_lps"].items()
    }
    report["model_xyz_continuous_index"] = {
        label: list(source_image.TransformPhysicalPointToContinuousIndex(point))
        for label, point in report["ali_corrected_physical_xyz_uncalibrated"].items()
    }
    report["drr_projection_uv"] = {
        label: list(project_drr_xyz(xyz))
        for label, xyz in report["model_xyz_continuous_index"].items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, default=DEFAULT_VOLUME)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ali_probe/04002")
    parser.add_argument("--ali-cli", type=Path, default=DEFAULT_ALI)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_MODELS.parent)
    parser.add_argument("--ali-python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--labels", nargs="+", default=["S", "N", "Ba"])
    parser.add_argument("--run", action="store_true", help="Run the optional upstream detector")
    parser.add_argument("--inspect-existing", action="store_true", help="Recompute coordinates from existing ALI output")
    args = parser.parse_args()

    for path in (args.volume,):
        if not path.is_file():
            parser.error(f"Missing volume: {path}")
    image = sitk.ReadImage(str(args.volume))
    values = sitk.GetArrayFromImage(image)
    if values.shape != (128, 128, 128) or not np.issubdtype(values.dtype, np.floating):
        parser.error(f"Expected SC-DREG 128^3 floating volume; got {values.shape} {values.dtype}")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1.001:
        parser.error("Expected finite SC-DREG intensities in [0, approximately 1]")
    if args.run and image.GetDirection() != (1., 0., 0., 0., 1., 0., 0., 0., 1.):
        parser.error("This probe supports only the SC-DREG identity NIfTI direction")
    native, native_lo, native_hi = native_histogram_cast(values)
    # Hypothetical CBCT-like contrast only: maps model air=0 to -1000 and
    # model high density=1 to +1000 before ALI's own histogram correction.
    proxy = (-1000 + 2000 * np.clip(values, 0, 1)).astype(np.float32)
    adapted, adapted_lo, adapted_hi = native_histogram_cast(proxy)
    report = {
        "source_volume": str(args.volume.resolve()),
        "source_sha256": sha256(args.volume),
        "source_shape_zyx": list(values.shape), "source_dtype": str(values.dtype),
        "source_range": [float(values.min()), float(values.max())],
        "source_spacing_xyz_uncalibrated": list(image.GetSpacing()),
        "source_origin_xyz": list(image.GetOrigin()),
        "source_direction": list(image.GetDirection()),
        "native_ali_cast": {"clip": [native_lo, native_hi], "nonzero_voxels": int(np.count_nonzero(native))},
        "intensity_hypothesis": "proxy_hu_linear = -1000 + 2000 * clip(sc_dreg, 0, 1); arbitrary contrast proxy, not HU",
        "adapted_ali_cast": {"clip": [adapted_lo, adapted_hi], "nonzero_voxels": int(np.count_nonzero(adapted))},
        "geometry_warning": "NIfTI unit spacing is uncalibrated; ALI treats it as physical spacing.",
        "projection_warning": "SC-DREG DRR geometry is synthetic and ALI output orientation may be incompatible.",
        "environment": {name: package_version(name) for name in ("torch", "monai", "itk", "SimpleITK", "numpy")},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = args.output_dir / "sc_dreg_proxy_hu.nii.gz"
    if args.run:
        if not args.ali_cli.is_file() or not args.ali_python.is_file():
            parser.error("ALI source checkout or isolated Python is missing; see docs/ali_cbct_probe.md")
        ensure_checkpoints(args.models_root, args.archive_dir, args.labels)
        checkpoints = {label: checkpoint_paths(args.models_root, label) for label in args.labels}
        report["checkpoints"] = {label: {scale: str(path.resolve()) for scale, path in scales.items()}
                                 for label, scales in checkpoints.items()}
        report["checkpoint_sha256"] = {label: {scale: sha256(path) for scale, path in scales.items()}
                                       for label, scales in checkpoints.items()}
        revision = subprocess.run(["git", "-C", str(args.ali_cli.parent.parent), "rev-parse", "HEAD"],
                                  text=True, capture_output=True, check=False)
        report["ali_source_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
        prep_image = sitk.GetImageFromArray(proxy)
        prep_image.CopyInformation(image)
        sitk.WriteImage(prep_image, str(prepared))
        cmd = [str(args.ali_python), str(args.ali_cli), str(prepared), str(args.models_root),
               ",".join(repr(label) for label in args.labels), str(args.output_dir),
               str(args.output_dir / "temp"), "false", "[1,0.3]", "[1,1]", "[64,64,64]", "10"]
        start_ns = time.time_ns()
        run = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        (args.output_dir / "ali.log").write_text(run.stdout, encoding="utf-8")
        report["ali_returncode"] = run.returncode
        report["ali_log"] = str(args.output_dir / "ali.log")
        report["ali_markups_lps"] = read_markups(args.output_dir, prepared.name.split(".")[0], start_ns)
        report["ali_version_note"] = "ALI SlicerAutomatedDentalTools CLI; revision must be recorded separately"
        add_coordinate_analysis(report, args.output_dir, prepared, image)
        if run.returncode or not set(args.labels).issubset(report["ali_markups_lps"]):
            print(run.stdout[-4000:], file=sys.stderr)
            print("ALI exited unsuccessfully or omitted requested landmarks; inspect ali.log", file=sys.stderr)
            status = 1
        else:
            status = 0
    elif args.inspect_existing:
        existing = args.output_dir / "probe.json"
        if not existing.is_file():
            parser.error(f"No prior ALI probe found: {existing}")
        report.update(json.loads(existing.read_text(encoding="utf-8")))
        report["ali_markups_lps"] = read_markups(args.output_dir, prepared.name.split(".")[0])
        report["checkpoint_sha256"] = {
            label: {scale: sha256(Path(path)) for scale, path in scales.items()}
            for label, scales in report.get("checkpoints", {}).items()
        }
        revision = subprocess.run(["git", "-C", str(args.ali_cli.parent.parent), "rev-parse", "HEAD"],
                                  text=True, capture_output=True, check=False)
        report["ali_source_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
        add_coordinate_analysis(report, args.output_dir, prepared, image)
        status = 0
    else:
        status = 0
    path = args.output_dir / "probe.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}; native nonzero={report['native_ali_cast']['nonzero_voxels']}, "
          f"adapted nonzero={report['adapted_ali_cast']['nonzero_voxels']}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
