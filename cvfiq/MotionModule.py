"""
MotionModule — Real-time motion detection using background subtraction.
No extra dependencies required.
"""

import cv2
import numpy as np


class MotionDetector:
    """
    Detect motion using MOG2 background subtraction.

    Usage:
        motion = MotionDetector(minArea=500)
        detected, regions, img = motion.findMotion(img)
        if detected:
            print(f"{len(regions)} moving regions")
    """

    def __init__(self, minArea=500, history=500, varThreshold=16, blurSize=21):
        """
        :param minArea:       Minimum contour area to count as motion (pixels²)
        :param history:       Number of frames used by background model
        :param varThreshold:  Variance threshold for background/foreground decision
        :param blurSize:      Gaussian blur kernel size (odd number) before subtraction
        """
        self.minArea    = minArea
        self.blurSize   = blurSize if blurSize % 2 == 1 else blurSize + 1
        self._bg        = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=varThreshold, detectShadows=False
        )
        self._kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def findMotion(self, img, draw=True):
        """
        Find motion regions in img.

        Returns:
            detected (bool): True if any motion found
            regions (list): [{"bbox": (x,y,w,h), "center": (cx,cy), "area": int}]
            img: annotated image
        """
        blurred = cv2.GaussianBlur(img, (self.blurSize, self.blurSize), 0)
        fgMask  = self._bg.apply(blurred)
        fgMask  = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, self._kernel)
        fgMask  = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN,  self._kernel)

        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.minArea:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            regions.append({"bbox": (x, y, w, h), "center": (cx, cy), "area": int(area)})
            if draw:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(img, f"{int(area)}px²", (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if draw and regions:
            cv2.putText(img, f"Motion: {len(regions)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return len(regions) > 0, regions, img

    def getMask(self, img):
        """Return the raw foreground mask (grayscale)."""
        blurred = cv2.GaussianBlur(img, (self.blurSize, self.blurSize), 0)
        return self._bg.apply(blurred)

    def reset(self):
        """Reset background model."""
        self._bg = cv2.createBackgroundSubtractorMOG2()
