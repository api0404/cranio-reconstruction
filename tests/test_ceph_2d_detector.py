"""Dependency-free tests for detector coordinate maps and landmark adapters."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from canonicalize_from_ceph_landmarks import FIT_LABELS, build_config
from ceph_canonicalization import fov_transform, geometry_diagnostics, map_points
from run_ceph_2d_detector import ceph_angles, detector_inputs, sha256, summary


class TwoDimensionalDetectorTests(unittest.TestCase):
    def test_detector_maps_round_trip_and_scope_of_anisotropic_resize(self):
        with tempfile.TemporaryDirectory() as folder:
            plans = detector_inputs(Image.new("L", (2808, 2136)), Path(folder))
            points = [[100, 200], [1300, 900], [2700, 2000]]
            for name, item in plans.items():
                forward = np.asarray(item["original_to_detector"])
                inverse = np.asarray(item["detector_to_original"])
                self.assertTrue(np.allclose(map_points(inverse, map_points(forward, points)), points))
                singular = item["linear_singular_values"]
                if name == "full_stretch":
                    self.assertNotAlmostEqual(*singular)
                else:
                    self.assertAlmostEqual(*singular)

    def test_angles_and_error_summary_are_derived_without_mm_claim(self):
        self.assertEqual(summary([1, 2, 3])["mean_px"], 2)
        angles = ceph_angles({"S": [0, 0], "N": [1, 0], "A": [1, 1], "B": [1, -1]})
        self.assertAlmostEqual(angles["SNA_degrees"], 90)
        self.assertAlmostEqual(angles["SNB_degrees"], 90)
        self.assertAlmostEqual(angles["ANB_degrees"], 0)

    def test_manual_and_automatic_marks_fit_same_explicit_template(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.png"
            Image.new("L", (500, 500), 100).save(source)
            source_hash = sha256(source)
            manual_points = {label: [float(50 + index * 60), float(80 + (index % 2) * 150)]
                             for index, label in enumerate(FIT_LABELS)}
            manual = root / "manual.json"
            manual.write_text(json.dumps({
                "coordinate_system": {"units": "source_image_pixels"},
                "landmarks": {"skeletal": [{"id": key, "x": xy[0], "y": xy[1]}
                                           for key, xy in manual_points.items()]}}))
            template_matrix = fov_transform([250, 250], 450, 2)
            template = root / "transforms.json"
            template.write_text(json.dumps({"source": {"sha256": source_hash},
                                            "candidates": [{"name": "tighter_fov",
                                                            "original_to_model": template_matrix.tolist()}]}))
            original_bytes = manual.read_bytes()
            config = build_config(source, manual, manual, template)
            self.assertEqual(manual.read_bytes(), original_bytes)
            self.assertEqual(config["provenance"]["landmark_kind"], "curated_manual")
            self.assertEqual([x["name"] for x in config["landmarks"]], list(FIT_LABELS))
            from canonicalize_ceph import candidates_from_config
            fitted = candidates_from_config(config, (500, 500), config["landmarks"])[0][1]
            self.assertTrue(np.allclose(fitted, template_matrix, atol=1e-12))
            for _, matrix, _ in candidates_from_config(config, (500, 500), config["landmarks"]):
                geometry_diagnostics(matrix, (500, 500))
            automatic = root / "automatic.json"
            shifted = {label: {"x": xy[0] + 3, "y": xy[1] - 2}
                       for label, xy in manual_points.items()}
            automatic.write_text(json.dumps({
                "kind": "automatic_pyceph_unreviewed", "source_image_sha256": source_hash,
                "coordinate_system": {"units": "source_image_pixels"}, "landmarks": shifted}))
            auto_config = build_config(source, automatic, manual, template)
            self.assertEqual(auto_config["provenance"]["landmark_kind"], "automatic_pyceph_unreviewed")
            self.assertEqual(manual.read_bytes(), original_bytes)
            automatic.write_text(automatic.read_text().replace(source_hash, "0" * 64))
            with self.assertRaisesRegex(ValueError, "different source image"):
                build_config(source, automatic, manual, template)


if __name__ == "__main__":
    unittest.main()
