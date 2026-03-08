"""
Test ColorModule.
Tracks a color in real time with optional HSV trackbar.
Press 't'=toggle trackbar, 'q'=quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

def main():
    print("=== ColorModule Test ===")
    print("  Point camera at a colored object.")
    print("  Controls: t=toggle trackbar  q=quit")

    trackbar_on = False
    cf = cvfiq.color(trackBar=trackbar_on)

    hsvVals = {'hmin': 35, 'smin': 80, 'vmin': 80,
               'hmax': 85, 'smax': 255, 'vmax': 255}

    with cvfiq.Camera(0, title="ColorModule Test") as cam:
        for img in cam:
            imgColor, mask = cf.update(img, hsvVals)

            status = "Trackbar: ON" if trackbar_on else "Trackbar: OFF  [t] toggle"
            cvfiq.putText(img, status, (10, 30),
                          cvfiq.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cvfiq.imshow("Original", img)
            cvfiq.imshow("Color Mask", mask)
            cvfiq.imshow("Color Result", imgColor)

            key = cvfiq.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                trackbar_on = not trackbar_on
                cvfiq.destroyAllWindows()
                cf = cvfiq.color(trackBar=trackbar_on)
                print(f"  [INFO] Trackbar {'ON' if trackbar_on else 'OFF'}")

    print("  PASSED")

if __name__ == "__main__":
    main()
