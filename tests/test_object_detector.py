"""
Test ObjectDetectorModule (MediaPipe Tasks API).
Model is auto-downloaded on first run.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

def main():
    print("=== ObjectDetectorModule Test ===")
    print("  Model auto-downloaded if not present.")
    print("  Point camera at objects (person, cup, phone…). Press q to quit.")

    detector = cvfiq.detector(maxResults=5, scoreThreshold=0.4)

    with cvfiq.Camera(0, showFPS=True, title="ObjectDetectorModule Test") as cam:
        for img in cam:
            objects, img = detector.findObjects(img)

            count  = detector.countObjects(objects)
            labels = list(set(o['label'] for o in objects))

            cvfiq.putText(img, f"Objects: {count}", (10, 30),
                          cvfiq.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if labels:
                cvfiq.putText(img, f"{', '.join(labels)}", (10, 60),
                              cvfiq.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            cam.show(img)

    print("  PASSED")

if __name__ == "__main__":
    main()
