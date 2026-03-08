"""
Test SelfiSegmentationModule.
Uses webcam — replaces background with solid color or blur.
Press 'c'=next color, 'b'=blur bg, 'q'=quit.
"""

import sys
import cv2
import numpy as np
sys.path.insert(0, '..')

from cvfiq.SelfiSegmentationModule import SelfiSegmentation
from cvfiq.FPS import FPS

COLORS = [
    ((0, 255, 0),   "Green"),
    ((255, 0, 0),   "Blue"),
    ((0, 0, 255),   "Red"),
    ((0, 255, 255), "Yellow"),
    ((255, 0, 255), "Magenta"),
    ((0, 0, 0),     "Black"),
    ((255, 255, 255), "White"),
]

def main():
    print("=== SelfiSegmentationModule Test ===")
    print("  Controls: c=next color  b=blur background  q=quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return

    seg = SelfiSegmentation()
    fps = FPS()
    mode = 'color'
    color_idx = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        if mode == 'color':
            bgr, name = COLORS[color_idx]
            imgOut = seg.removeBG(img, imgBg=bgr, threshold=0.7)
            label = f"Color: {name}"
        else:
            blurred = cv2.GaussianBlur(img, (55, 55), 0)
            imgOut = seg.removeBG(img, imgBg=blurred, threshold=0.7)
            label = "Blur BG"

        f = fps.update()

        cv2.putText(imgOut, f"FPS: {f:.1f}  {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(imgOut, "c=next color  b=blur  q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("SelfiSegmentation Test", imgOut)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            mode = 'blur'
            print("  [INFO] Mode -> Blur")
        elif key == ord('c'):
            mode = 'color'
            color_idx = (color_idx + 1) % len(COLORS)
            print(f"  [INFO] Color -> {COLORS[color_idx][1]}")

    cap.release()
    cv2.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
