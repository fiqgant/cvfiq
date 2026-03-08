"""
Test FaceLandmarkerModule (MediaPipe Tasks API).
Model will be downloaded automatically if not present.
Press 'q' to quit.
"""

import sys
import os
import cv2
sys.path.insert(0, '..')

from cvfiq.FaceLandmarkerModule import FaceLandmarker
from cvfiq.FPS import FPS
from _download_model import download_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

def main():
    print("=== FaceLandmarkerModule Test ===")

    if not download_model(MODEL_URL, MODEL_PATH):
        print("  [ERROR] Could not get model, exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return

    detector = FaceLandmarker(MODEL_PATH, maxFaces=1, outputBlendshapes=True)
    fps = FPS()

    print("  Try smiling, blinking, opening your mouth.")
    print("  Press q to quit.")

    while True:
        success, img = cap.read()
        if not success:
            break

        faces, img = detector.findFaces(img)
        f = fps.update()

        if faces:
            expr        = detector.getExpression(faces[0])
            smiling     = detector.isSmiling(faces[0])
            blink_left  = detector.isBlinking(faces[0], eye='left')
            blink_right = detector.isBlinking(faces[0], eye='right')
            mouth       = detector.isMouthOpen(faces[0])

            lines = [
                f"Expression: {expr}",
                f"Smiling:    {'YES' if smiling else 'no'}",
                f"L-Blink:    {'YES' if blink_left else 'no'}",
                f"R-Blink:    {'YES' if blink_right else 'no'}",
                f"Mouth open: {'YES' if mouth else 'no'}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(img, line, (10, 60 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        cv2.putText(img, f"FPS: {f:.1f}  Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("FaceLandmarkerModule Test — q to quit", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
