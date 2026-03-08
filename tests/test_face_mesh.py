"""
Test FaceMeshModule.
Shows 468 landmarks, blink detection, mouth open detection.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

def main():
    print("=== FaceMeshModule Test ===")
    print("  Face the camera. Try blinking and opening your mouth. Press q to quit.")

    detector = cvfiq.mesh(maxFaces=1, refineLandmarks=True)

    with cvfiq.Camera(0, showFPS=True, title="FaceMeshModule Test") as cam:
        for img in cam:
            img, faces = detector.findFaceMesh(img, draw=True)

            if faces:
                blink = detector.blinkDetector(faces[0])
                mouth = detector.mouthOpen(faces[0])

                cvfiq.putText(img, f"L-eye: {'BLINK' if not blink['left'] else 'open'}",
                              (10, 60), cvfiq.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cvfiq.putText(img, f"R-eye: {'BLINK' if not blink['right'] else 'open'}",
                              (10, 90), cvfiq.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cvfiq.putText(img, f"Mouth: {'OPEN' if mouth['open'] else 'closed'} ({mouth['ratio']:.2f})",
                              (10, 120), cvfiq.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)

            cam.show(img)

    print("  PASSED")

if __name__ == "__main__":
    main()
