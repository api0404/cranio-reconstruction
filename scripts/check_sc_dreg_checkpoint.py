"""Strictly load published checkpoint into the network without PCA/volumes."""

from collections import Counter
from pathlib import Path

import torch

from sc_dreg_compat import ROOT, load_model_class


class NetworkOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        upstream = load_model_class().__init__.__globals__
        self.reg23d = upstream["Reg23D"](60)
        self.refinenet = upstream["RefineNet"]()
        self.unet = upstream["UNet"](1, 2, 4)


def main() -> None:
    path = ROOT / "models" / "cbct_c2f_model_ckpt.tar"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model = NetworkOnly()
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    state = checkpoint["model_state_dict"]
    print(f"PyTorch {torch.__version__}; CUDA available={torch.cuda.is_available()}; CUDA build={torch.version.cuda}")
    print(f"Checkpoint epoch={checkpoint['epoch']}; tensors={len(state)}; dtypes={dict(Counter(str(v.dtype) for v in state.values()))}")
    print(f"Network parameters={sum(p.numel() for p in model.parameters())}; {result}")
    unet_path = ROOT / "models" / "xray_seg_unet_ckpt.tar"
    if unet_path.is_file():
        unet = torch.load(unet_path, map_location="cpu", weights_only=True, mmap=True)
        print(f"Standalone UNet checkpoint epoch={unet['epoch']}; "
              f"{model.unet.load_state_dict(unet['model_state_dict'], strict=True)}")
        differences = [key for key, value in unet["model_state_dict"].items()
                       if not torch.equal(value, state["unet." + key])]
        print(f"Standalone versus embedded UNet: equal={len(unet['model_state_dict']) - len(differences)}, "
              f"different={len(differences)}, changed fields={dict(Counter(key.split('.')[-1] for key in differences))}")


if __name__ == "__main__":
    main()
