"""Small image-conditioned BC baseline for LIBERO (torch, optional).

This is a runnable baseline, not a Diffusion Policy reimplementation: it gives the LIBERO track a
complete train/eval path and a checkpoint format. Swap in DP/ACT by keeping the same
``fit/save/load/act`` interface.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ImageBC:
    def __init__(self, action_dim: int = 7, device: str = "cpu"):
        self.action_dim = int(action_dim)
        self.device = device
        self.model = None

    def _ensure(self):
        if self.model is not None:
            return
        try:
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("ImageBC needs torch: pip install torch") from exc
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(),
            nn.Linear(64 * 6 * 6, 256), nn.ReLU(),
            nn.Linear(256, self.action_dim),
        ).to(self.device)

    def _to_tensor(self, images: np.ndarray):
        import torch
        x = np.asarray(images, dtype=np.float32) / 255.0
        if x.ndim == 3:
            x = x[None]
        return torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)

    def fit(self, images: np.ndarray, actions: np.ndarray, epochs: int = 5,
            batch: int = 64, lr: float = 1e-3) -> list[float]:
        import torch
        self._ensure()
        X = self._to_tensor(images)
        Y = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        for _ in range(int(epochs)):
            perm = torch.randperm(len(X), device=self.device)
            total = 0.0
            for i in range(0, len(X), batch):
                idx = perm[i:i + batch]
                loss = torch.nn.functional.mse_loss(self.model(X[idx]), Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            losses.append(total / len(X))
        self.model.eval()
        return losses

    def act(self, image: np.ndarray) -> np.ndarray:
        import torch
        self._ensure()
        with torch.no_grad():
            return self.model(self._to_tensor(image)).cpu().numpy()[0]

    def save(self, path: str | Path) -> None:
        import torch
        self._ensure()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "action_dim": self.action_dim}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ImageBC":
        import torch
        blob = torch.load(path, map_location=device)
        obj = cls(action_dim=int(blob["action_dim"]), device=device)
        obj._ensure()
        obj.model.load_state_dict(blob["state_dict"])
        obj.model.eval()
        return obj
