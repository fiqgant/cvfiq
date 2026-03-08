"""
Test ArucoModule.
Generates a marker then detects it from webcam.
Press 'g'=show generated marker, 'q'=quit.
"""

import sys
sys.path.insert(0, '..')

import cvfiq

def main():
    print("=== ArucoModule Test ===")
    print("  Controls: g=show marker image  q=quit")

    detector = cvfiq.aruco(dictType='DICT_4X4_50')

    marker = detector.generateMarker(markerId=1, size=300)
    cvfiq.imwrite("aruco_marker_1.png", marker)
    print("  Generated aruco_marker_1.png — show it to the camera!")

    with cvfiq.Camera(0, showFPS=True, title="ArucoModule Test") as cam:
        for img in cam:
            markers, img = detector.findMarkers(img, draw=True)

            for m in markers:
                cx, cy = m['center']
                cvfiq.putText(img, f"ID:{m['id']}", (cx - 20, cy - 10),
                              cvfiq.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cvfiq.putText(img, f"Markers: {len(markers)}  |  g=show marker  q=quit",
                          (10, 30), cvfiq.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cvfiq.imshow("ArucoModule Test", img)
            key = cvfiq.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                big = cvfiq.resize(marker, (600, 600), interpolation=cvfiq.INTER_NEAREST)
                cvfiq.imshow("ArUco Marker ID=1 (show to camera)", big)

    print("  PASSED")

if __name__ == "__main__":
    main()
