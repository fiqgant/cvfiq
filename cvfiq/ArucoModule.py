"""
ArUco Marker Detection Module
Uses OpenCV aruco module (built-in since OpenCV 4.7+).
No extra model files needed.
"""

import cv2
import numpy as np


class ArucoDetector:
    """
    Detects ArUco markers and estimates their 3D pose.
    ArUco markers are square black-and-white patterns used for AR, robotics,
    camera calibration, and position tracking.
    """

    DICT_TYPES = {
        '4x4_50':   cv2.aruco.DICT_4X4_50,
        '4x4_100':  cv2.aruco.DICT_4X4_100,
        '4x4_250':  cv2.aruco.DICT_4X4_250,
        '5x5_50':   cv2.aruco.DICT_5X5_50,
        '5x5_100':  cv2.aruco.DICT_5X5_100,
        '6x6_50':   cv2.aruco.DICT_6X6_50,
        '6x6_100':  cv2.aruco.DICT_6X6_100,
    }

    def __init__(self, dictType='4x4_50'):
        """
        :param dictType: ArUco dictionary type (e.g. '4x4_50', '5x5_100')
        """
        arucoDict = cv2.aruco.getPredefinedDictionary(
            self.DICT_TYPES.get(dictType, cv2.aruco.DICT_4X4_50)
        )
        arucoParams = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        self._dict = arucoDict

    def findMarkers(self, img, draw=True):
        """
        Detect ArUco markers in a BGR image.
        :param img: Input BGR image
        :param draw: Draw marker outlines, centers, and IDs
        :return: list of marker dicts, image
                 Each dict: {"id", "corners": [[x,y]x4], "center": (cx,cy)}
        """
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(imgGray)

        markers = []
        if ids is not None:
            for i, corner in enumerate(corners):
                pts = corner[0].astype(int)
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())
                markerInfo = {
                    "id": int(ids[i][0]),
                    "corners": pts.tolist(),
                    "center": (cx, cy),
                }
                markers.append(markerInfo)

                if draw:
                    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, f'ID:{markerInfo["id"]}',
                                (pts[0][0], pts[0][1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        if draw:
            return markers, img
        return markers

    def estimatePose(self, img, markers, markerSize=0.05,
                     cameraMatrix=None, distCoeffs=None):
        """
        Estimate 3D pose (rotation + translation) of each detected marker.
        :param img: Input image (for drawing axes)
        :param markers: List from findMarkers()
        :param markerSize: Physical size of the marker in meters
        :param cameraMatrix: Camera intrinsic matrix (auto-estimated if None)
        :param distCoeffs: Distortion coefficients (zeros if None)
        :return: list of pose dicts, image
                 Each dict: {"id", "rvec", "tvec", "distance"}
        """
        h, w, _ = img.shape
        if cameraMatrix is None:
            cameraMatrix = np.array([[w, 0, w / 2],
                                     [0, w, h / 2],
                                     [0, 0, 1]], dtype=np.float32)
        if distCoeffs is None:
            distCoeffs = np.zeros((4, 1), dtype=np.float32)

        half = markerSize / 2
        objPoints = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

        poses = []
        for marker in markers:
            imgPoints = np.array(marker["corners"], dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(
                objPoints, imgPoints, cameraMatrix, distCoeffs)
            if success:
                distance = float(np.linalg.norm(tvec))
                cv2.drawFrameAxes(img, cameraMatrix, distCoeffs,
                                  rvec, tvec, markerSize * 0.5)
                cv2.putText(img, f'{distance:.2f}m',
                            (marker["center"][0], marker["center"][1] + 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
                poses.append({
                    "id": marker["id"],
                    "rvec": rvec,
                    "tvec": tvec,
                    "distance": distance,
                })

        return poses, img

    def generateMarker(self, markerId, size=200, dictType='4x4_50'):
        """
        Generate an ArUco marker image.
        :param markerId: Marker ID to generate
        :param size: Output image size in pixels
        :param dictType: Dictionary type
        :return: Marker image (grayscale)
        """
        arucoDict = cv2.aruco.getPredefinedDictionary(
            self.DICT_TYPES.get(dictType, cv2.aruco.DICT_4X4_50)
        )
        markerImg = cv2.aruco.generateImageMarker(arucoDict, markerId, size)
        return markerImg


def main():
    cap = cv2.VideoCapture(0)
    detector = ArucoDetector(dictType='4x4_50')
    while True:
        success, img = cap.read()
        markers, img = detector.findMarkers(img)
        if markers:
            poses, img = detector.estimatePose(img, markers, markerSize=0.05)
            for pose in poses:
                print(f"ID:{pose['id']}  dist:{pose['distance']:.3f}m")
        cv2.imshow("ArUco", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
