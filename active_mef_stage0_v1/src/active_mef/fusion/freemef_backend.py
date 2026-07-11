from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

from .base import FusionBackend


class FreeMEFBackend(FusionBackend):
    """Adapter around the official external FreeMEF repository.

    FreeMEF recurrently consumes the auxiliary frames, so their acquisition
    order is semantically relevant. The configured base exposure is kept as the
    main frame and the remaining selected exposures are passed in caller order.
    """

    order_sensitive = True

    def __init__(
        self,
        repo_path: str,
        checkpoint: str,
        config: str,
        device: str = "cuda",
        checkpoint_key: str = "params",
        input_color: str = "bgr",
        base_ev: float = 0.0,
        pad_factor: int = 8,
    ) -> None:
        import torch
        import yaml

        self.torch = torch
        self.device = torch.device(device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu")
        repo = Path(repo_path).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from FreeMEF.archs.FreeMEF_arch import FreeMEF

        cfg_path = Path(config)
        with cfg_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        net_cfg = dict(cfg["network_g"])
        net_cfg.pop("type", None)
        self.model = FreeMEF(**net_cfg)
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt[checkpoint_key] if isinstance(ckpt, dict) and checkpoint_key in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.input_color = input_color.lower()
        if self.input_color not in {"rgb", "bgr"}:
            raise ValueError("input_color must be 'rgb' or 'bgr'")
        self.base_ev = float(base_ev)
        self.pad_factor = int(pad_factor)
        if self.pad_factor <= 0:
            raise ValueError("pad_factor must be positive")

    def _tensor(self, img: np.ndarray):
        if self.input_color == "bgr":
            img = img[..., ::-1].copy()
        tensor = self.torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        return tensor.unsqueeze(0).to(self.device)

    def _pad(self, base, others):
        import torch.nn.functional as F

        h, w = base.shape[-2:]
        pad_h = (self.pad_factor - h % self.pad_factor) % self.pad_factor
        pad_w = (self.pad_factor - w % self.pad_factor) % self.pad_factor
        if pad_h == 0 and pad_w == 0:
            return base, others, h, w
        mode = "reflect" if h > pad_h and w > pad_w else "replicate"
        base = F.pad(base, (0, pad_w, 0, pad_h), mode=mode)
        if others is not None:
            batch, count, channels, oh, ow = others.shape
            flat = others.reshape(batch * count, channels, oh, ow)
            flat = F.pad(flat, (0, pad_w, 0, pad_h), mode=mode)
            others = flat.reshape(batch, count, channels, h + pad_h, w + pad_w)
        return base, others, h, w

    def fuse(self, exposures: dict[float, np.ndarray], selected_evs: list[float]) -> np.ndarray:
        torch = self.torch
        selected = [float(e) for e in selected_evs]
        if not selected:
            raise ValueError("selected_evs must be non-empty")
        missing = [ev for ev in selected if ev not in exposures]
        if missing:
            raise KeyError(f"selected EVs missing from exposure pool: {missing}")

        base_ev = self.base_ev if self.base_ev in selected else selected[0]
        base = self._tensor(exposures[base_ev])
        others_evs = []
        removed_base = False
        for ev in selected:
            if ev == base_ev and not removed_base:
                removed_base = True
                continue
            others_evs.append(ev)

        if others_evs:
            others = torch.stack(
                [self._tensor(exposures[ev]).squeeze(0) for ev in others_evs],
                dim=0,
            ).unsqueeze(0)
            count = torch.tensor([len(others_evs)], dtype=torch.long, device=self.device)
        else:
            others = None
            count = None

        base, others, original_h, original_w = self._pad(base, others)
        with torch.inference_mode():
            out = self.model(base, others, count)
        out = out[..., :original_h, :original_w]
        arr = out.clamp(0, 1)[0].detach().cpu().numpy().transpose(1, 2, 0)
        if self.input_color == "bgr":
            arr = arr[..., ::-1]
        return arr.astype(np.float32)
