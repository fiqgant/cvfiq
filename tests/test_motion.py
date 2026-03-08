"""
Test MotionModule — background subtraction motion detection.
Requires webcam. Move in front of the camera.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq


def main():
    print("=== Motion Detector Test ===")
    print("  Move in front of the camera to trigger motion.")
    print("  Press 'q' to quit.")

    detector = cvfiq.motion(minArea=500)

    with cvfiq.Camera(0, showFPS=True, title="Motion Test") as cam:
        for img in cam:
            # draw=True already annotates bounding boxes and "Motion: N" label
            detected, regions, img = detector.findMotion(img)
            cam.show(img)

    print("  PASSED")


if __name__ == "__main__":
    main()
