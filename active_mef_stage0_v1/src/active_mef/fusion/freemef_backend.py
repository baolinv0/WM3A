from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

from .base import FusionBackend


class FreeMEFBackend(FusionBackend):
    """Thin adapter around the official external FreeMEF repository.

    No FreeMEF source code is copied here. Supply repo_path, checkpoint, and the
    official YAML config. This wrapper follows the released forward contract:
    model(ldr, others, others_count).
    """

    def __init__(
        self,
        repo_path: str,
        checkpoint: str,
        config: str,
        device: str = "cuda",
        checkpoint_key: str = "params",
        input_color: str = "bgr",
    ) -> None:
        import torch
        import yaml

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        repo = Path(repo_path).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from FreeMEF.archs.FreeMEF_arch import FreeMEF

        cfg_path = Path(config)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        net_cfg = dict(cfg["network_g"])
        net_cfg.pop("type", None)
        self.model = FreeMEF(**net_cfg)
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt[checkpoint_key] if isinstance(ckpt, dict) and checkpoint_key in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.input_color = input_color.lower()

    def _tensor(self, img: np.ndarray):
        if self.input_color == "bgr":
            img = img[..., ::-1].copy()
        t = self.torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        return t.unsqueeze(0).to(self.device)

    def fuse(self, exposures: dict[float, np.ndarray], selected_evs: list[float]) -> np.ndarray:
        torch = self.torch
        selected = sorted(float(e) for e in selected_evs)
        base_idx = len(selected) // 2
        base_ev = selected[base_idx]
        base = self._tensor(exposures[base_ev])
        others_evs = [e for i, e in enumerate(selected) if i != base_idx]
        if others_evs:
            others = torch.stack([self._tensor(exposures[e]).squeeze(0) for e in others_evs], dim=0).unsqueeze(0)
            count = torch.tensor([len(others_evs)], device=self.device)
        else:
            others = None
            count = None
        with torch.no_grad():
            out = self.model(base, others, count)
        arr = out.clamp(0, 1)[0].detach().cpu().numpy().transpose(1, 2, 0)
        if self.input_color == "bgr":
            arr = arr[..., ::-1]
        return arr.astype(np.float32)
