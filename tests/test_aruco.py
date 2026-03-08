"""
Test ArucoModule.
Generates a marker image then detects it from webcam.
Press 'g'=generate & show marker, 'q'=quit.
"""

import sys
import cv2
import numpy as np
sys.path.insert(0, '..')

from cvfiq.ArucoModule import ArucoDetector
from cvfiq.FPS import FPS

def main():
    print("=== ArucoModule Test ===")
    print("  Show a printed/displayed ArUco marker to the camera.")
    print("  Controls: g=generate marker image  q=quit")

    detector = ArucoDetector(dictType='DICT_4X4_50')
    fps = FPS()

    # Generate and save a sample marker
    marker = detector.generateMarker(markerId=1, size=300)
    cv2.imwrite("aruco_marker_1.png", marker)
    print("  Generated aruco_marker_1.png — show it to the camera!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera")
        return

    while True:
        success, img = cap.read()
        if not success:
            break

        markers, img = detector.findMarkers(img, draw=True)
        f = fps.update()

        if markers:
            for m in markers:
                mid = m['id']
                cx, cy = m['center']
                cv2.putText(img, f"ID:{mid}", (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(img, f"FPS: {f:.1f}  Markers: {len(markers)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, "g=show marker  q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("ArucoModule Test", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            big = cv2.resize(marker, (600, 600), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("ArUco Marker ID=1 (show to camera)", big)

    cap.release()
    cv2.destroyAllWindows()
    print("  PASSED")

if __name__ == "__main__":
    main()
