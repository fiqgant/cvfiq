"""
Test SelfiSegmentationModule.
Replaces background with solid color or blur.
Press 'c'=next color, 'b'=blur bg, 'q'=quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq
import numpy as np

COLORS = [
    ((0, 255, 0),     "Green"),
    ((255, 0, 0),     "Blue"),
    ((0, 0, 255),     "Red"),
    ((0, 255, 255),   "Yellow"),
    ((255, 0, 255),   "Magenta"),
    ((0, 0, 0),       "Black"),
    ((255, 255, 255), "White"),
]

def main():
    print("=== SelfiSegmentationModule Test ===")
    print("  Controls: c=next color  b=blur background  q=quit")

    seg       = cvfiq.segment()
    mode      = 'color'
    color_idx = 0

    with cvfiq.Camera(0, title="SelfiSegmentation Test") as cam:
        for img in cam:
            if mode == 'color':
                bgr, name = COLORS[color_idx]
                imgOut = seg.removeBG(img, imgBg=bgr, smooth=True)
                label  = f"Color: {name}"
            else:
                blurred = cvfiq.GaussianBlur(img, (55, 55), 0)
                imgOut  = seg.removeBG(img, imgBg=blurred, smooth=True)
                label   = "Blur BG"

            cvfiq.putText(imgOut, f"{label}  |  c=color  b=blur  q=quit",
                          (10, 30), cvfiq.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cvfiq.imshow("SelfiSegmentation Test", imgOut)
            key = cvfiq.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                mode = 'blur'
                print("  [INFO] Mode -> Blur")
            elif key == ord('c'):
                mode = 'color'
                color_idx = (color_idx + 1) % len(COLORS)
                print(f"  [INFO] Color -> {COLORS[color_idx][1]}")

    print("  PASSED")

if __name__ == "__main__":
    main()
