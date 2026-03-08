"""
Test PoseModule.
Shows 33 body landmarks and joint angles.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

def main():
    print("=== PoseModule Test ===")
    print("  Stand in front of the camera. Press q to quit.")

    detector = cvfiq.pose(smooth=True)

    with cvfiq.Camera(0, showFPS=True, title="PoseModule Test") as cam:
        for img in cam:
            img = detector.findPose(img)
            lmList, bboxInfo = detector.findPosition(img)

            if lmList:
                angle = detector.findAngle(img, 12, 14, 16)
                cvfiq.putText(img, f"Right elbow: {angle:.1f} deg", (10, 60),
                              cvfiq.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

                dist, img, _ = detector.findDistance(23, 24, img)
                cvfiq.putText(img, f"Hip width: {dist:.0f}px", (10, 90),
                              cvfiq.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

            cam.show(img)

    print("  PASSED")

if __name__ == "__main__":
    main()
