"""
Test ObjectDetectorModule (MediaPipe Tasks API).
Model will be downloaded automatically if not present.
Press 'q' to quit.
"""

import sys
import os
import cv2
sys.path.insert(0, '..')

from cvfiq.ObjectDetectorModule import ObjectDetector
from cvfiq.FPS import FPS
from _download_model import download_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'efficientdet_lite0.tflite')
MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite'

def main():
    print("=== ObjectDetectorModule Test ===")

    if not download_model(MODEL_URL, MODEL_PATH):
        print("  [ERROR] Could not get model, exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return

    detector = ObjectDetector(MODEL_PATH, maxResults=5, scoreThreshold=0.4)
    fps = FPS()

    print("  Point camera at objects (person, cup, phone, etc.).")
    print("  Press q to quit.")

    while True:
        success, img = cap.read()
        if not success:
            break

        objects, img = detector.findObjects(img)
        f = fps.update()

        count = detector.countObjects(objects)
        cv2.putText(img, f"FPS: {f:.1f}  Objects: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if objects:
            labels = list(set(o['label'] for o in objects))
            cv2.putText(img, f"Detected: {', '.join(labels)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        cv2.imshow("ObjectDetectorModule Test — q to quit", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
