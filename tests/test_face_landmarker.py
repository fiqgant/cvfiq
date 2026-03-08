"""
Test FaceLandmarkerModule (MediaPipe Tasks API).
Model is auto-downloaded on first run.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq
from cvfiq.FaceLandmarkerModule import FaceLandmarker

def main():
    print("=== FaceLandmarkerModule Test ===")
    print("  Model auto-downloaded if not present.")
    print("  Try smiling, blinking, opening your mouth. Press q to quit.")

    detector = cvfiq.landmarker(maxFaces=1, outputBlendshapes=True)

    with cvfiq.Camera(0, showFPS=True, title="FaceLandmarkerModule Test") as cam:
        for img in cam:
            faces, img = detector.findFaces(img)

            if faces:
                expr   = detector.getExpression(faces[0])
                smile  = detector.isSmiling(faces[0])
                bl     = detector.isBlinking(faces[0], eye='left')
                br     = detector.isBlinking(faces[0], eye='right')
                mouth  = detector.isMouthOpen(faces[0])
                jaw    = detector.getBlendshape(faces[0], FaceLandmarker.JAW_OPEN)

                lines = [
                    f"Expression: {expr}",
                    f"Smiling:    {'YES' if smile else 'no'}",
                    f"L-Blink:    {'YES' if bl else 'no'}",
                    f"R-Blink:    {'YES' if br else 'no'}",
                    f"Mouth open: {'YES' if mouth else 'no'}  ({jaw:.2f})",
                ]
                for i, line in enumerate(lines):
                    cvfiq.putText(img, line, (10, 60 + i * 28),
                                  cvfiq.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            cam.show(img)

    print("  PASSED")

if __name__ == "__main__":
    main()
