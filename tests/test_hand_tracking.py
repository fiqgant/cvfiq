"""
Test HandTrackingModule.
Shows hand landmarks, finger count, distance, and angle.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

def main():
    print("=== HandTrackingModule Test ===")
    print("  Show your hand to the camera. Press q to quit.")

    detector = cvfiq.hand(maxHands=2, detectionCon=0.8)

    with cvfiq.Camera(0, showFPS=True, title="HandTrackingModule Test") as cam:
        for img in cam:
            hands, img = detector.findHands(img)

            if hands:
                hand1  = hands[0]
                lmList = hand1['lmList']
                fingers = detector.fingersUp(hand1)

                cvfiq.putText(img, f"Fingers: {fingers.count(1)}", (10, 60),
                              cvfiq.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if len(lmList) >= 9:
                    dist, info, img = detector.findDistance(lmList[4], lmList[8], img)
                    cvfiq.putText(img, f"Thumb-Index: {dist:.0f}px", (10, 100),
                                  cvfiq.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

                if len(lmList) >= 17:
                    angle, _ = detector.findAngle(lmList[8], lmList[12], lmList[16], img)
                    cvfiq.putText(img, f"Angle 8-12-16: {angle:.1f}", (10, 135),
                                  cvfiq.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

            cam.show(img)

    print("  PASSED")

if __name__ == "__main__":
    main()
