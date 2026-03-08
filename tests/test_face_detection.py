"""
Test FaceDetectionModule.
Uses webcam — shows bounding box, score, and 6 keypoints.
Press 'q' to quit.
"""

import sys
import cv2
sys.path.insert(0, '..')

from cvfiq.FaceDetectionModule import FaceDetector
from cvfiq.FPS import FPS

def main():
    print("=== FaceDetectionModule Test ===")
    print("  Face the camera.")
    print("  Press q to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return

    detector = FaceDetector(minDetectionCon=0.75)
    fps = FPS()

    while True:
        success, img = cap.read()
        if not success:
            break

        img, bboxs = detector.findFaces(img)
        f = fps.update()

        if bboxs:
            for bbox in bboxs:
                score = bbox.get('score', [0])[0]
                kp = bbox.get('keypoints', {})
                cv2.putText(img, f"Conf: {score:.2f}", (bbox['bbox'][0], bbox['bbox'][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                for name, pt in kp.items():
                    cv2.circle(img, pt, 4, (0, 0, 255), -1)

        cv2.putText(img, f"FPS: {f:.1f}  Faces: {len(bboxs)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("FaceDetectionModule Test — q to quit", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
