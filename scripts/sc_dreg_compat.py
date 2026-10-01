"""Narrow, in-memory compatibility edits to the unmodified SC-DREG model."""

from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True  # Keep the upstream submodule clean on import.


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SRC = ROOT / "vendor" / "sc-dreg" / "src"


def load_model_class():
    """Load upstream c2f_model with device and memory changes only."""
    import torch

    source_path = UPSTREAM_SRC / "model.py"
    source = source_path.read_text(encoding="utf-8")
    replacements = {
        "net = models.resnet34(pretrained=True)": "net = models.resnet34(weights=None)",
        "self.param_path = param_path": "self.param_path = param_path\n        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "self.COEFF = torch.Tensor(np.load(os.path.join(param_path, 'coeff4.npy'))).cuda()":
            "self.COEFF = torch.from_numpy(np.load(os.path.join(param_path, 'coeff4.npy'), mmap_mode='r')[:pca_dim].astype(np.float32)).to(device)",
        "self.mean_ = torch.Tensor(np.load(os.path.join(param_path, 'mean4.npy'))).cuda()":
            "self.mean_ = torch.from_numpy(np.load(os.path.join(param_path, 'mean4.npy'), mmap_mode='r').astype(np.float32)).to(device)",
        ".cuda().reshape(1, 1, 128, 128, 128)": ".to(device).reshape(1, 1, 128, 128, 128)",
        "self.mesh = torch.from_numpy(mesh).cuda().float()": "self.mesh = torch.from_numpy(mesh).to(device).float()",
        "calibs = torch.eye(4).unsqueeze(0).cuda()": "calibs = torch.eye(4, device=points.device).unsqueeze(0)",
    }
    for old, new in replacements.items():
        expected = 2 if old == ".cuda().reshape(1, 1, 128, 128, 128)" else 1
        actual = source.count(old)
        if actual != expected:
            raise RuntimeError(f"Upstream model.py changed: expected {expected} occurrence(s) of {old!r}, found {actual}")
        source = source.replace(old, new)

    sys.path.insert(0, str(UPSTREAM_SRC))
    module = types.ModuleType("sc_dreg_compat_model")
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module.c2f_model
