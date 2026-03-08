"""
Test GestureModule (MediaPipe Tasks API).
Model is auto-downloaded on first run.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

def main():
    print("=== GestureModule Test ===")
    print("  Model auto-downloaded if not present.")
    print("  Show hand gestures: Thumb_Up, Victory, Open_Palm, etc. Press q to quit.")

    detector = cvfiq.gesture()

    with cvfiq.Camera(0, showFPS=True, title="GestureModule Test") as cam:
        for img in cam:
            gestures, img = detector.findGestures(img)

            for i, g in enumerate(gestures):
                cvfiq.putText(img, f"{g['hand']}: {g['gesture']} ({g['score']:.0%})",
                              (10, 60 + i * 35),
                              cvfiq.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            cam.show(img)

    print("  PASSED")

if __name__ == "__main__":
    main()
