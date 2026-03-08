"""
Video Stabilizer Module
Real-time video stabilization using optical flow (Lucas-Kanade).
No model files needed — pure OpenCV.
"""

import cv2
import numpy as np


class VideoStabilizer:
    """
    Real-time video stabilization using sparse optical flow.
    Reduces shaky/jitter motion from handheld cameras.
    """

    def __init__(self, smoothRadius=15, maxCorners=200, border='black'):
        """
        :param smoothRadius: Number of frames to average for smoothing.
                             Higher = smoother but more lag
        :param maxCorners: Number of feature points to track
        :param border: Border fill mode: 'black', 'replicate', or 'reflect'
        """
        self.smoothRadius = smoothRadius
        self.maxCorners = maxCorners
        self.border = border

        self._prevGray = None
        self._prevTransform = np.zeros(3, dtype=np.float64)
        self._trajectory = []

        self._borderModes = {
            'black':     cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
            'reflect':   cv2.BORDER_REFLECT,
        }

    def stabilize(self, img):
        """
        Stabilize a single frame. Call on every frame in order.
        :param img: Input BGR image
        :return: Stabilized BGR image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self._prevGray is None:
            self._prevGray = gray
            self._trajectory.append(self._prevTransform.copy())
            return img

        # Detect feature points to track
        prevPts = cv2.goodFeaturesToTrack(
            self._prevGray,
            maxCorners=self.maxCorners,
            qualityLevel=0.01,
            minDistance=30,
        )

        if prevPts is None or len(prevPts) < 5:
            self._prevGray = gray
            self._trajectory.append(self._prevTransform.copy())
            return img

        # Track points with optical flow
        currPts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prevGray, gray, prevPts, None)

        idx = np.where(status == 1)[0]
        if len(idx) < 5:
            self._prevGray = gray
            self._trajectory.append(self._prevTransform.copy())
            return img

        prevPts = prevPts[idx]
        currPts = currPts[idx]

        # Estimate affine transform (translation + rotation + scale)
        m, _ = cv2.estimateAffinePartial2D(prevPts, currPts)
        if m is None:
            self._prevGray = gray
            self._trajectory.append(self._prevTransform.copy())
            return img

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        # Accumulate trajectory
        self._prevTransform = self._prevTransform + np.array([dx, dy, da])
        self._trajectory.append(self._prevTransform.copy())

        # Smooth trajectory with moving average window
        n = len(self._trajectory)
        start = max(0, n - self.smoothRadius)
        smoothed = np.mean(self._trajectory[start:n], axis=0)

        # Compute correction needed
        correction = smoothed - self._prevTransform
        dx_c, dy_c, da_c = correction

        cos_a = np.cos(da_c)
        sin_a = np.sin(da_c)
        m_smooth = np.array([
            [cos_a, -sin_a, dx_c],
            [sin_a,  cos_a, dy_c],
        ], dtype=np.float32)

        h, w = img.shape[:2]
        border_mode = self._borderModes.get(self.border, cv2.BORDER_CONSTANT)
        imgStab = cv2.warpAffine(img, m_smooth, (w, h),
                                 borderMode=border_mode)

        self._prevGray = gray
        return imgStab

    def reset(self):
        """Reset stabilizer state. Call when switching video sources."""
        self._prevGray = None
        self._prevTransform = np.zeros(3, dtype=np.float64)
        self._trajectory = []

    def getSmoothness(self):
        """
        Estimate current smoothness score (0=very shaky, 1=very smooth).
        Based on variance of recent trajectory.
        :return: float 0.0 to 1.0
        """
        if len(self._trajectory) < 2:
            return 1.0
        recent = self._trajectory[-min(30, len(self._trajectory)):]
        variance = float(np.mean(np.var(recent, axis=0)))
        return round(max(0.0, 1.0 - min(1.0, variance / 100.0)), 3)


def main():
    cap = cv2.VideoCapture(0)
    stabilizer = VideoStabilizer(smoothRadius=15)

    while True:
        success, img = cap.read()
        if not success:
            break

        imgStab = stabilizer.stabilize(img)
        smoothness = stabilizer.getSmoothness()

        cv2.putText(imgStab, f'Smooth: {smoothness:.2f}',
                    (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        combined = np.hstack([img, imgStab])
        cv2.putText(combined, 'Original', (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(combined, 'Stabilized', (img.shape[1] + 10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Video Stabilizer", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
