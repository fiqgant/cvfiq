"""
Test GestureModule (MediaPipe Tasks API).
Model will be downloaded automatically if not present.
Press 'q' to quit.
"""

import sys
import os
import cv2
sys.path.insert(0, '..')

from cvfiq.GestureModule import GestureDetector
from cvfiq.FPS import FPS
from _download_model import download_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'gesture_recognizer.task')
MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'

def main():
    print("=== GestureModule Test ===")

    if not download_model(MODEL_URL, MODEL_PATH):
        print("  [ERROR] Could not get model, exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return

    detector = GestureDetector(MODEL_PATH)
    fps = FPS()

    print("  Show hand gestures: Thumb_Up, Victory, Open_Palm, etc.")
    print("  Press q to quit.")

    while True:
        success, img = cap.read()
        if not success:
            break

        gestures, img = detector.findGestures(img)
        f = fps.update()

        if gestures:
            for i, g in enumerate(gestures):
                name  = g['gesture']
                score = g['score']
                hand  = g['hand']
                cv2.putText(img, f"{hand}: {name} ({score:.0%})",
                            (10, 60 + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.putText(img, f"FPS: {f:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("GestureModule Test — q to quit", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
