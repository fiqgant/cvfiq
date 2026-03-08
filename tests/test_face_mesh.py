"""
Test FaceMeshModule.
Uses webcam — shows 468 landmarks, blink detection, mouth open detection.
Press 'q' to quit.
"""

import sys
import cv2
sys.path.insert(0, '..')

from cvfiq.FaceMeshModule import FaceMeshDetector
from cvfiq.FPS import FPS

def main():
    print("=== FaceMeshModule Test ===")
    print("  Face the camera. Try blinking and opening your mouth.")
    print("  Press q to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return

    detector = FaceMeshDetector(maxFaces=1, refineLandmarks=True)
    fps = FPS()

    while True:
        success, img = cap.read()
        if not success:
            break

        img, faces = detector.findFaceMesh(img, draw=True)
        f = fps.update()

        if faces:
            face = faces[0]

            blink = detector.blinkDetector(face)
            mouth = detector.mouthOpen(face)

            left_blink  = "BLINK" if not blink['left']  else "open"
            right_blink = "BLINK" if not blink['right'] else "open"
            mouth_state = "OPEN" if mouth['open'] else "closed"

            cv2.putText(img, f"Left eye:  {left_blink}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(img, f"Right eye: {right_blink}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(img, f"Mouth: {mouth_state} (MAR={mouth['ratio']:.2f})", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)

        cv2.putText(img, f"FPS: {f:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("FaceMeshModule Test — q to quit", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
