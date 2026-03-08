"""
Test HandTrackingModule.
Uses webcam — shows hand landmarks, finger count, distance, angle.
Press 'q' to quit.
"""

import sys
import cv2
sys.path.insert(0, '..')

from cvfiq.HandTrackingModule import HandDetector
from cvfiq.FPS import FPS

def main():
    print("=== HandTrackingModule Test ===")
    print("  Show your hand to the camera.")
    print("  Press q to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return

    detector = HandDetector(maxHands=2, detectionCon=0.8)
    fps = FPS()

    while True:
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img)
        f = fps.update()

        if hands:
            hand = hands[0]
            lmList = hand['lmList']
            fingers = detector.fingersUp(hand)
            fingerCount = fingers.count(1)

            cv2.putText(img, f"Fingers: {fingerCount}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if len(lmList) >= 9:
                dist, info, _ = detector.findDistance(lmList[4], lmList[8], img)
                cv2.putText(img, f"Thumb-Index dist: {dist:.0f}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

            if len(lmList) >= 17:
                angle, _ = detector.findAngle(lmList[8], lmList[12], lmList[16], img)
                cv2.putText(img, f"Angle 8-12-16: {angle:.1f}", (10, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

        cv2.putText(img, f"FPS: {f:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("HandTrackingModule Test — q to quit", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
