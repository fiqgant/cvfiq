import cv2
import mediapipe as mp
import math


class FaceMeshDetector:
    """
    Face Mesh Detector to find 468 Landmarks using the mediapipe library.
    Helps acquire the landmark points in pixel format
    """

    def __init__(self, staticMode=False, maxFaces=2, refineLandmarks=False,
                 minDetectionCon=0.5, minTrackCon=0.5):
        """
        :param staticMode: In static mode, detection is done on each image: slower
        :param maxFaces: Maximum number of faces to detect
        :param refineLandmarks: Refine landmarks around eyes and lips (478 points vs 468)
        :param minDetectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLandmarks = refineLandmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                 max_num_faces=self.maxFaces,
                                                 refine_landmarks=self.refineLandmarks,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        Finds face landmarks in BGR Image.
        :param img: Image to find the face landmarks in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

    # Landmark index groups for facial regions
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
    RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]

    def getRegion(self, face, region):
        """
        Get landmark points for a specific facial region.
        :param face: Face landmark list from findFaceMesh
        :param region: List of landmark indices (use class constants e.g. FaceMeshDetector.LEFT_EYE)
        :return: List of (x, y) points for the region
        """
        return [face[i] for i in region if i < len(face)]

    def blinkDetector(self, face):
        """
        Detect if eyes are open or closed using Eye Aspect Ratio (EAR).
        :param face: Face landmark list from findFaceMesh
        :return: dict with 'left', 'right' booleans (True = open), and EAR values
        """
        def ear(eye_top, eye_bottom, eye_left, eye_right):
            vertical = math.hypot(eye_top[0] - eye_bottom[0], eye_top[1] - eye_bottom[1])
            horizontal = math.hypot(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1])
            return vertical / horizontal if horizontal != 0 else 0

        left_ear = ear(face[159], face[145], face[133], face[33])
        right_ear = ear(face[386], face[374], face[362], face[263])
        threshold = 0.2
        return {
            "left": left_ear > threshold,
            "right": right_ear > threshold,
            "leftEAR": round(left_ear, 3),
            "rightEAR": round(right_ear, 3)
        }

    def mouthOpen(self, face):
        """
        Detect if the mouth is open using Mouth Aspect Ratio (MAR).
        :param face: Face landmark list from findFaceMesh
        :return: dict with 'open' boolean and 'ratio' value
        """
        vertical = math.hypot(face[13][0] - face[14][0], face[13][1] - face[14][1])
        horizontal = math.hypot(face[78][0] - face[308][0], face[78][1] - face[308][1])
        mar = vertical / horizontal if horizontal != 0 else 0
        return {"open": mar > 0.15, "ratio": round(mar, 3)}

    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if faces:
            print(faces[0])
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
