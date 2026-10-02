"""Check comparison geometry and array metrics without private assets or inference."""

from pathlib import Path
import sys
import json
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ceph_canonicalization import fov_transform, translation
from compare_manual_auto_sc_dreg import pair_metrics, transform_parameters, validate_controlled_configs


class ManualAutoComparisonTests(unittest.TestCase):
    def test_crop_center_and_rotation_come_from_inverse_matrix(self):
        matrix = fov_transform([180, 230], 800, 3.5)
        parameters = transform_parameters(matrix)
        np.testing.assert_allclose(parameters["crop_center_source_xy"], [180, 230], atol=1e-10)
        self.assertAlmostEqual(parameters["rotation_degrees_image_xy"], 3.5)
        self.assertAlmostEqual(parameters["source_fov_width_px"], 800)
        shifted = transform_parameters(translation(2, -4) @ matrix)
        expected_source_shift = -np.linalg.inv(matrix[:2, :2]) @ np.array([2, -4])
        np.testing.assert_allclose(np.asarray(shifted["crop_center_source_xy"])-[180, 230],
                                   expected_source_shift)

    def test_pair_metrics_reports_known_difference(self):
        a = np.array([[0, 1], [2, 3]], dtype=np.float32)
        b = a + 2
        result = pair_metrics(a, b)
        self.assertAlmostEqual(result["mae"], 2)
        self.assertAlmostEqual(result["rmse"], 2)
        self.assertAlmostEqual(result["ncc"], 1)

    def test_control_rejects_non_landmark_preprocessing_change(self):
        base = {"source": "original.png", "target": "04002.png", "orientation": {},
                "intensity": {"mode": "identity"}, "artifact_masks": [], "candidates": [],
                "provenance": {"manual_template_sha256": "m", "template_transform_sha256": "t",
                               "fit_labels": ["S", "N", "Me"]},
                "landmarks": [{"name": "S", "role": "fit", "target": [1, 2], "source": [3, 4]}]}
        with tempfile.TemporaryDirectory() as folder:
            manual, automatic = Path(folder) / "manual", Path(folder) / "automatic"
            manual.mkdir()
            automatic.mkdir()
            (manual / "generated_config.json").write_text(json.dumps(base))
            changed = json.loads(json.dumps(base))
            changed["landmarks"][0]["source"] = [5, 6]
            (automatic / "generated_config.json").write_text(json.dumps(changed))
            validate_controlled_configs(manual, automatic)
            changed["intensity"] = {"mode": "invert"}
            (automatic / "generated_config.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "configs differ"):
                validate_controlled_configs(manual, automatic)


if __name__ == "__main__":
    unittest.main()
