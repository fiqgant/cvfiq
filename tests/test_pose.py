"""
Test PoseModule.
Uses webcam — shows 33 body landmarks, angle between joints.
Press 'q' to quit.
"""

import sys
import cv2
sys.path.insert(0, '..')

from cvfiq.PoseModule import PoseDetector
from cvfiq.FPS import FPS

def main():
    print("=== PoseModule Test ===")
    print("  Stand in front of the camera (full body visible is best).")
    print("  Press q to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return

    detector = PoseDetector(smooth=True)
    fps = FPS()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img)
        f = fps.update()

        if lmList:
            # Elbow angle: shoulder(12) - elbow(14) - wrist(16)
            angle = detector.findAngle(img, 12, 14, 16)
            cv2.putText(img, f"Right elbow: {angle:.1f} deg", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

            # Hip distance
            dist, img, _ = detector.findDistance(23, 24, img)
            cv2.putText(img, f"Hip width: {dist:.0f}px", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

            cv2.putText(img, f"Landmarks: {len(lmList)}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.putText(img, f"FPS: {f:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("PoseModule Test — q to quit", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
