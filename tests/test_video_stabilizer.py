"""
Test VideoStabilizerModule.
Shows original vs stabilized video side by side from webcam.
Controls: q=quit  r=reset  s=cycle radius  b=cycle border
"""

import sys
import cv2
import numpy as np
sys.path.insert(0, '..')

from cvfiq.VideoStabilizerModule import VideoStabilizer
from cvfiq.FPS import FPS

SMOOTH_RADII = [5, 15, 30]
BORDER_MODES = ['black', 'replicate', 'reflect']

def main():
    print("=== VideoStabilizerModule Test ===")
    print("  Move the camera around to see stabilization.")
    print("  Controls: q=quit  r=reset  s=cycle radius  b=cycle border")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return

    smooth_idx = 1
    border_idx = 0
    stab = VideoStabilizer(smoothRadius=SMOOTH_RADII[smooth_idx],
                           border=BORDER_MODES[border_idx])
    fps = FPS(avgCount=20)

    while True:
        success, img = cap.read()
        if not success:
            break

        imgStab = stab.stabilize(img)
        smoothness = stab.getSmoothness()
        f = fps.update()
        H, W = img.shape[:2]

        info = [
            f"FPS: {f:.1f}",
            f"Smoothness: {smoothness:.3f}",
            f"Radius: {SMOOTH_RADII[smooth_idx]}",
            f"Border: {BORDER_MODES[border_idx]}",
        ]
        for i, line in enumerate(info):
            cv2.putText(imgStab, line, (10, 30 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        bar_w = int(smoothness * (W - 20))
        bar_color = (0, int(255 * smoothness), int(255 * (1 - smoothness)))
        cv2.rectangle(imgStab, (10, H - 20), (W - 10, H - 8), (50, 50, 50), -1)
        cv2.rectangle(imgStab, (10, H - 20), (10 + bar_w, H - 8), bar_color, -1)

        cv2.putText(img, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(imgStab, "Stabilized", (10, H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        divider = np.full((H, 4, 3), 255, dtype=np.uint8)
        combined = np.hstack([img, divider, imgStab])

        cv2.imshow("VideoStabilizer Test  [q=quit | r=reset | s=radius | b=border]", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            stab.reset()
            print("  [INFO] Stabilizer reset")
        elif key == ord('s'):
            smooth_idx = (smooth_idx + 1) % len(SMOOTH_RADII)
            stab.smoothRadius = SMOOTH_RADII[smooth_idx]
            print(f"  [INFO] smoothRadius -> {SMOOTH_RADII[smooth_idx]}")
        elif key == ord('b'):
            border_idx = (border_idx + 1) % len(BORDER_MODES)
            stab.border = BORDER_MODES[border_idx]
            print(f"  [INFO] border -> {BORDER_MODES[border_idx]}")

    cap.release()
    cv2.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
