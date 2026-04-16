from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mef_pipeline import prepare_model_inputs


class Backend:
    """
    Template adapter for plugging a real MEF-Net model into run_mef_dataset.py.

    Replace the placeholder logic in fuse() with your actual model import,
    checkpoint loading, preprocessing, inference, and postprocessing.
    """

    def __init__(self, model_path: Path | None = None, device: str | None = None) -> None:
        self.model_path = model_path
        self.device = device or "cpu"

        # Example:
        # import torch
        # from your_mefnet_package import MEFNet
        # self.model = MEFNet().to(self.device)
        # state = torch.load(self.model_path, map_location=self.device)
        # self.model.load_state_dict(state)
        # self.model.eval()

    def fuse(self, images, bracket_set, model_path=None, device=None) -> np.ndarray:
        """
        images: list of BGR uint8 numpy arrays from the Pi bracket capture
        bracket_set: metadata describing the set_id, labels, timestamps, etc.
        """
        stack = prepare_model_inputs(images, rgb=True)

        # Example:
        # import torch
        # tensor = torch.from_numpy(stack).permute(0, 3, 1, 2).unsqueeze(0)
        # tensor = tensor.to(self.device)
        # with torch.inference_mode():
        #     fused = self.model(tensor)
        # fused_rgb = fused.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Placeholder: average the stack so the adapter runs before MEF-Net is wired in.
        fused_rgb = np.mean(stack, axis=0)
        fused_rgb = np.clip(fused_rgb * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(fused_rgb, cv2.COLOR_RGB2BGR)
