"""
DepthModule — Monocular depth estimation using MiDaS via OpenCV DNN.
Model is auto-downloaded on first use (~25 MB).
No extra dependencies required beyond opencv-python.
"""

import cv2
import numpy as np
import os
import urllib.request


_MODEL_URL  = (
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
)
_MODEL_FILE = "midas_small.onnx"
_INPUT_SIZE = (256, 256)
_MEAN       = (123.675, 116.28, 103.53)
_STD        = (58.395, 57.12, 57.375)


class DepthEstimator:
    """
    Estimate depth from a single RGB image using MiDaS (small, fast).

    Usage:
        depth = DepthEstimator()
        depthMap, colorized = depth.findDepth(img)   # colorized is a nice heatmap
        dist = depth.getDistance(depthMap, (cx, cy))  # relative depth at point
    """

    def __init__(self, modelPath=_MODEL_FILE, colormap=cv2.COLORMAP_MAGMA):
        """
        :param modelPath: Path to MiDaS ONNX file (auto-downloaded if missing)
        :param colormap:  cv2.COLORMAP_* for the colorized output
        """
        self.colormap = colormap
        if not os.path.exists(modelPath):
            self._download(modelPath)
        self._net = cv2.dnn.readNet(modelPath)

    def _download(self, path):
        print(f"[cvfiq] Downloading depth model ({path}) ...")
        def _prog(c, b, t):
            pct = min(c * b * 100 // t, 100)
            print(f"\r[cvfiq] {path}: {pct}%", end='', flush=True)
        urllib.request.urlretrieve(_MODEL_URL, path, reporthook=_prog)
        print()

    def findDepth(self, img):
        """
        Estimate depth for img.

        Returns:
            depthMap (np.ndarray float32): raw depth values (higher = closer in MiDaS)
            colorized (np.ndarray uint8 BGR): false-color depth map, same size as img
        """
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / 255.0, _INPUT_SIZE,
            mean=_MEAN, swapRB=True, crop=False
        )
        self._net.setInput(blob)
        out = self._net.forward()   # shape: (1, 1, H, W)
        depth = out[0, 0]           # (H_out, W_out)

        h, w = img.shape[:2]
        depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        depth_norm = cv2.normalize(depth_resized, None, 0, 255,
                                   cv2.NORM_MINMAX, cv2.CV_8U)
        colorized = cv2.applyColorMap(depth_norm, self.colormap)

        return depth_resized, colorized

    def getDistance(self, depthMap, point):
        """
        Get relative depth value at pixel (x, y).
        Higher value = closer to camera (MiDaS convention).
        """
        h, w = depthMap.shape[:2]
        x = int(min(max(point[0], 0), w - 1))
        y = int(min(max(point[1], 0), h - 1))
        return float(depthMap[y, x])

    def overlay(self, img, alpha=0.5):
        """Return img with depth heatmap blended on top."""
        _, colorized = self.findDepth(img)
        return cv2.addWeighted(img, 1 - alpha, colorized, alpha, 0)
