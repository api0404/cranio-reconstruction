"""Small geometry and field checks for the exploration exports."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from explore_sc_dreg_reference import mesh_at_level  # noqa: E402
from explore_sc_dreg_deformation import capture, magnitude_summary  # noqa: E402


class ExplorationTests(unittest.TestCase):
    def test_mesh_coordinates_are_xyz_from_zyx_volume(self):
        volume = np.zeros((8, 9, 10), dtype=np.float32)
        volume[2:5, 3:7, 4:8] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "box.ply"
            result = mesh_at_level(volume, .5, path)
            self.assertEqual(result["bounds_xyz"], [[3.5, 2.5, 1.5], [7.5, 6.5, 4.5]])
            self.assertGreater(result["triangles"], 0)
            self.assertIn(b"format binary_little_endian 1.0", path.read_bytes()[:100])

    def test_capture_rejects_wrong_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, pca=np.zeros(60), coarse=np.zeros((2, 3)), refined=np.zeros((2, 3)))
            with self.assertRaisesRegex(ValueError, "Invalid deformation capture"):
                capture(path)

    def test_magnitude_and_mask(self):
        field = np.array([[[3, 4, 0], [0, 0, 12]]], dtype=np.float32)
        self.assertEqual(magnitude_summary(field)["max"], 12)
        self.assertEqual(magnitude_summary(field, np.array([[True, False]]))["rms"], 5)


if __name__ == "__main__":
    unittest.main()
