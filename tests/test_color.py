"""
Test ColorModule.
Uses webcam to pick and track a color in real time.
Press 't' to toggle trackbar, 'q' to quit.
"""

import sys
import cv2
sys.path.insert(0, '..')

from cvfiq.ColorModule import ColorFinder
from cvfiq.FPS import FPS

def main():
    print("=== ColorModule Test ===")
    print("  Point camera at a colored object.")
    print("  Controls: t=toggle trackbar  q=quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return

    trackbar_on = False
    cf = ColorFinder(trackBar=trackbar_on)
    fps = FPS()

    # Default: track green
    hsvVals = {'hmin': 35, 'smin': 80, 'vmin': 80,
               'hmax': 85, 'smax': 255, 'vmax': 255}

    while True:
        success, img = cap.read()
        if not success:
            break

        imgColor, mask = cf.update(img, hsvVals)
        f = fps.update()

        status = "Trackbar: ON (tune HSV)" if trackbar_on else "Trackbar: OFF  [t] to toggle"
        cv2.putText(img, f"FPS: {f:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Original", img)
        cv2.imshow("Color Mask", mask)
        cv2.imshow("Color Result", imgColor)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            trackbar_on = not trackbar_on
            cv2.destroyAllWindows()
            cf = ColorFinder(trackBar=trackbar_on)
            print(f"  [INFO] Trackbar {'ON' if trackbar_on else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
