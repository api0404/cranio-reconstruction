"""Small geometry and failure-mode tests; no ALI weights are required."""

from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ali_cbct_probe import (checkpoint_paths, correct_ali_export_xyz, ensure_checkpoints,
                            native_histogram_cast, project_drr_xyz)
from analyze_ali_cbct_probe import map_xy


class AliProbeTests(unittest.TestCase):
    def test_native_histogram_cast_erases_unit_intensity(self):
        volume = np.linspace(0, 1, 1000, dtype=np.float32).reshape(10, 10, 10)
        cast, _, high = native_histogram_cast(volume)
        self.assertLess(high, 1)
        self.assertEqual(np.count_nonzero(cast), 0)

    def test_drr_projection_uses_depth_dependent_uniform_scale(self):
        self.assertEqual(project_drr_xyz((63.5, 63.5, 63.5)), (63.5, 63.5))
        left = project_drr_xyz((0, 73.5, 63.5))
        right = project_drr_xyz((127, 73.5, 63.5))
        self.assertLess(left[0], right[0])
        self.assertEqual(left[1], 63.5)

    def test_projection_matches_upstream_drr_sampling_grid(self):
        import torch
        from torch.nn import functional as functional

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/sc-dreg/src"))
        prior_bytecode_setting = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # Keep the pinned upstream checkout clean.
        try:
            from drr import GenerateDRR
        finally:
            sys.dont_write_bytecode = prior_bytecode_setting

        x, y, z = 64, 72, 80
        volume = torch.zeros((1, 1, 128, 128, 128))
        volume[0, 0, z, y, x] = 1
        sampled = functional.grid_sample(volume, GenerateDRR().tgrid, align_corners=False)
        image = torch.rot90(sampled.sum(2)[0, 0], 2, (0, 1))
        peak_v, peak_u = torch.nonzero(image == image.max())[0].tolist()
        predicted_u, predicted_v = project_drr_xyz((x, y, z))
        self.assertLess(abs(peak_u - predicted_u), 1)
        self.assertLess(abs(peak_v - predicted_v), 1)

    def test_missing_checkpoint_reports_label_and_scale(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "S checkpoint for scale 1"):
                checkpoint_paths(Path(temp), "S")

    def test_extracts_only_selected_checkpoint_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with zipfile.ZipFile(root / "Cranial_Base.zip", "w") as archive:
                for label in ("S", "N"):
                    for scale in ("1", "0-3"):
                        archive.writestr(f"Cranial_Base/{label}/{scale}/{label}_Net_{scale}.pth", b"weights")
            ensure_checkpoints(root / "weights", root, ["S"])
            self.assertEqual(len(checkpoint_paths(root / "weights", "S")), 2)
            self.assertFalse((root / "weights/Cranial_Base/N").exists())

    def test_missing_archive_names_required_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(FileNotFoundError, "Cranial_Base.zip"):
                ensure_checkpoints(root / "weights", root, ["S"])

    def test_ali_positive_resampling_origin_and_ceph_round_trip(self):
        self.assertTrue(np.allclose(correct_ali_export_xyz([65.25, 76, 85.5], (.1, .1, .1)),
                                    [65.45, 76.2, 85.7]))
        matrix = np.array([[.05, 0, -12], [0, .05, -8], [0, 0, 1.]])
        point = [1612.4, 685.5]
        self.assertTrue(np.allclose(map_xy(np.linalg.inv(matrix), map_xy(matrix, point)), point))


if __name__ == "__main__":
    unittest.main()
