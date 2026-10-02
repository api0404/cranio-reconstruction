"""Regenerate an explicitly approximate unmasked subject input from published project notes.

The previous private transform/mask files are missing in this checkout. The
rounded values below come from docs/manual_auto_sc_dreg_comparison.md and do
not reproduce that prior manual input exactly.
"""

import argparse
import json
from pathlib import Path

from canonicalize_ceph import ROOT, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/private/Cephalometric_X-Ray.png")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/documented_subject_approx")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "source": str(args.source.resolve()),
        "orientation": {"flip_horizontal": False, "flip_vertical": False},
        "intensity": {"mode": "identity"}, "artifact_masks": [],
        "candidates": [{"name": "subject_approx_unmasked", "mode": "explicit_fov",
                        "center_oriented": [1799.64, 1099.68], "fov_width_px": 1603.57,
                        "rotation_degrees": 1.0502}],
        "provenance": {"status": "approximate_unmasked_not_prior_manual_anchor_tighter",
                       "source_report": "docs/manual_auto_sc_dreg_comparison.md",
                       "parameters_are_rounded_report_values": True,
                       "prior_case_specific_ruler_mask_unavailable": True}}
    config_path = args.output_dir / "generated_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    run(config_path, args.output_dir)


if __name__ == "__main__":
    main()
