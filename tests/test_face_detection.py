"""
Test FaceDetectionModule.
Shows bounding box, score, and 6 keypoints.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

def main():
    print("=== FaceDetectionModule Test ===")
    print("  Face the camera. Press q to quit.")

    detector = cvfiq.face(minDetectionCon=0.75)

    with cvfiq.Camera(0, showFPS=True, title="FaceDetectionModule Test") as cam:
        for img in cam:
            img, bboxs = detector.findFaces(img)

            if bboxs:
                for bbox in bboxs:
                    score = bbox.get('score', [0])[0]
                    kp    = bbox.get('keypoints', {})
                    cvfiq.putText(img, f"Conf: {score:.2f}",
                                  (bbox['bbox'][0], bbox['bbox'][1] - 10),
                                  cvfiq.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    for pt in kp.values():
                        cvfiq.circle(img, pt, 4, (0, 0, 255), -1)

            cvfiq.putText(img, f"Faces: {len(bboxs)}", (10, 30),
                          cvfiq.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cam.show(img)

    print("  PASSED")

if __name__ == "__main__":
    main()
