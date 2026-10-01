"""Geometry and local output checks for clinical-ceph candidate generation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ceph_canonicalization import (fit_similarity, fov_transform, geometry_diagnostics,
                                    map_points, orientation_matrix,
                                    pillow_inverse_coefficients, scale_about_model_center)
from canonicalize_ceph import apply_artifact_masks, run
from study_sc_dreg_input_sensitivity import ncc, sweep_transforms


class CanonicalizationTests(unittest.TestCase):
    def test_landmark_fit_and_inverse_with_explicit_flip(self):
        source = np.array([[12., 24.], [38., 29.], [17., 50.], [75., 64.]])
        flip = orientation_matrix(100, 80, True, False)
        expected = fov_transform([45, 39], 110, 3.5, flip)
        target = map_points(expected, source)
        fitted = fit_similarity(map_points(flip, source), target) @ flip
        self.assertTrue(np.allclose(fitted, expected, atol=1e-12))
        self.assertTrue(np.allclose(map_points(np.linalg.inv(fitted), target), source, atol=1e-12))
        self.assertLess(geometry_diagnostics(fitted, (100, 80))["determinant"], 0)

    def test_fov_and_scale_are_uniform_without_shear(self):
        matrix = scale_about_model_center(fov_transform([250, 300], 500, -8), 0.85)
        self.assertAlmostEqual(geometry_diagnostics(matrix, (700, 700))["source_fov_width_px"], 500 / 0.85)
        shear = matrix.copy()
        shear[0, 1] += 0.02
        with self.assertRaisesRegex(ValueError, "Non-uniform"):
            geometry_diagnostics(shear, (700, 700))

    def test_pillow_inverse_samples_pixel_centers(self):
        matrix = fov_transform([63.5, 63.5], 128)
        self.assertTrue(np.array_equal(np.array(pillow_inverse_coefficients(matrix)),
                                       np.array([1, 0, 0, 0, 1, 0])))
        image = Image.fromarray(np.arange(128 * 128, dtype=np.uint16).reshape(128, 128).astype("uint8"))
        result = image.transform((128, 128), Image.Transform.AFFINE,
                                 pillow_inverse_coefficients(matrix), resample=Image.Resampling.NEAREST)
        self.assertTrue(np.array_equal(np.asarray(result), np.asarray(image)))

    def test_cli_outputs_round_trip_metadata_without_changing_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "original.png"
            Image.fromarray(np.full((200, 240), 127, dtype=np.uint8)).save(original)
            before = original.read_bytes()
            config = root / "config.json"
            config.write_text(json.dumps({"source": str(original), "orientation": {"flip_horizontal": False},
                                          "candidates": [{"name": "wide", "mode": "explicit_fov",
                                                          "center_oriented": [120, 100], "fov_width_px": 180}],
                                          "landmarks": [{"name": "S", "source": [100, 90], "role": "check"}]}))
            output = root / "output"
            run(config, output)
            self.assertEqual(original.read_bytes(), before)
            with Image.open(output / "wide_128.png") as rendered:
                self.assertEqual(rendered.size, (128, 128))
            report = json.loads((output / "transforms.json").read_text())
            candidate = report["candidates"][0]
            self.assertEqual(candidate["landmarks"][0]["round_trip_original"], [100.0, 90.0])
            self.assertTrue(candidate["geometry"]["all_fov_corners_inside_source"])
            self.assertEqual(candidate["geometry"]["fraction_model_pixel_centers_outside_source"], 0)

    def test_artifact_mask_changes_only_explicit_region_and_rejects_landmark_overlap(self):
        image = Image.fromarray(np.full((20, 30), 120, dtype=np.uint8))
        region = {"name": "ruler", "polygon_original": [[22, 2], [27, 2], [27, 12], [22, 12]],
                  "fill_value": 0, "non_anatomical_reason": "test background ruler"}
        masked, mask, records = apply_artifact_masks(image, [region],
                                                     [{"name": "S", "source": [10, 10]}])
        self.assertEqual(np.asarray(image)[5, 24], 120)
        self.assertEqual(np.asarray(masked)[5, 24], 0)
        self.assertEqual(np.asarray(masked)[5, 10], 120)
        self.assertEqual(np.count_nonzero(np.asarray(masked) != np.asarray(image)),
                         np.count_nonzero(np.asarray(mask)))
        self.assertEqual(records[0]["name"], "ruler")
        with self.assertRaisesRegex(ValueError, "overlaps landmark"):
            apply_artifact_masks(image, [region], [{"name": "S", "source": [24, 5]}])

    def test_sweep_perturbations_are_small_similarity_transforms(self):
        base = fov_transform([100, 100], 180)
        transforms = dict(sweep_transforms(base))
        self.assertEqual(len(transforms), 8)
        for matrix in transforms.values():
            geometry_diagnostics(matrix, (220, 220))
        point = map_points(base, [[100, 100]])
        self.assertTrue(np.allclose(map_points(transforms["x_plus2px"], [[100, 100]]), point + [2, 0]))
        self.assertAlmostEqual(ncc(np.arange(10), np.arange(10)), 1)


if __name__ == "__main__":
    unittest.main()
