"""Checks that remain runnable while the large PCA basis is unavailable."""

from pathlib import Path
from contextlib import redirect_stdout
import io
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.dont_write_bytecode = True
from run_sc_dreg_demo import ASSET_NAMES, save_png, validate_inputs  # noqa: E402
from sc_dreg_compat import load_pca_basis  # noqa: E402


class BaselineTests(unittest.TestCase):
    def test_missing_asset_is_reported_before_numpy_or_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "coeff4.npy"):
                validate_inputs(Path(directory), ROOT / "vendor/sc-dreg/tests/04002.png", 60)

    def test_incomplete_npy_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            for name in ASSET_NAMES:
                (models / name).touch()
            with self.assertRaisesRegex(ValueError, "possibly incomplete download"):
                validate_inputs(models, ROOT / "vendor/sc-dreg/tests/04002.png", 60)

    def test_shape_check_uses_memory_mapping_only(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            for name in ASSET_NAMES:
                (models / name).touch()
            coeff = np.empty((0, 0), dtype=np.float64)
            mean = np.empty((0,), dtype=np.float64)
            with patch("run_sc_dreg_demo.np.load", side_effect=[coeff, mean]) as load:
                with self.assertRaisesRegex(ValueError, "Unexpected coeff4.npy shape"):
                    validate_inputs(models, ROOT / "vendor/sc-dreg/tests/04002.png", 60)
            self.assertEqual(load.call_count, 2)
            for call in load.call_args_list:
                self.assertEqual(call.kwargs, {"mmap_mode": "r", "allow_pickle": False})

    def test_pca_header_validation_without_allocating_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            for name in ASSET_NAMES:
                (models / name).touch()
            coeff = SimpleNamespace(shape=(80, 128**3 * 3), ndim=2, dtype=np.dtype("float64"))
            mean = SimpleNamespace(shape=(128**3 * 3,), dtype=np.dtype("float64"))
            with patch("run_sc_dreg_demo.np.load", side_effect=[coeff, mean]) as load:
                with redirect_stdout(io.StringIO()):
                    validate_inputs(models, ROOT / "vendor/sc-dreg/tests/04002.png", 60)
            self.assertEqual(load.call_count, 2)
            with self.assertRaisesRegex(ValueError, "60-output PCA head"):
                validate_inputs(models, ROOT / "vendor/sc-dreg/tests/04002.png", 61)

    def test_pca_slice_matches_upstream_float32_cast(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basis.npy"
            original = np.arange(40, dtype=np.float64).reshape(10, 4) / 7
            np.save(path, original)
            actual = load_pca_basis(path, 2, torch.device("cpu"))
            expected = torch.Tensor(original[:2])
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(tuple(actual.shape), (2, 4))

    def test_input_tensor_and_png_match_upstream(self):
        sys.path.insert(0, str(ROOT / "vendor/sc-dreg/src"))
        from utils import read_img

        source = ROOT / "vendor/sc-dreg/tests/04002.png"
        upstream = read_img(str(source))
        runner = torch.from_numpy(np.asarray(Image.open(source).convert("L"), dtype=np.float32) / 255.0)
        self.assertTrue(torch.equal(upstream, runner))
        self.assertEqual(tuple(runner.shape), (128, 128))
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "raw.png"
            save_png(generated, runner)
            self.assertTrue(np.array_equal(np.asarray(Image.open(generated)),
                                           np.asarray(Image.open(ROOT / "vendor/sc-dreg/tests/output/04002_raw.png"))))

    def test_large_and_private_assets_are_ignored(self):
        for path in ("models/coeff4.npy", "models/cbct_c2f_model_ckpt.tar",
                     "data/private/patient.png", "data/raw/scan.dcm", "outputs/run.nii.gz"):
            result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, check=False)
            self.assertEqual(result.returncode, 0, path)
        tracked = subprocess.check_output(["git", "ls-files", "--cached"], cwd=ROOT, text=True).splitlines()
        for path in tracked:
            self.assertFalse(path.startswith(("models/", "data/private/", "data/raw/", "outputs/"))
                             and not path.endswith(".gitkeep"), path)


if __name__ == "__main__":
    unittest.main()
