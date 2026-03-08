"""
Face Landmarker Module
Uses MediaPipe Tasks API FaceLandmarker.
Provides 478 landmarks + 52 blendshapes (facial expressions).
Requires model file: face_landmarker.task
Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
"""

import cv2
import mediapipe as mp


class FaceLandmarker:
    """
    Detects face landmarks and blendshape expressions using MediaPipe Tasks API.
    Returns 478 landmarks and 52 blendshape scores representing facial expressions.
    """

    # Common blendshape names for quick reference
    SMILE_LEFT = 'mouthSmileLeft'
    SMILE_RIGHT = 'mouthSmileRight'
    BLINK_LEFT = 'eyeBlinkLeft'
    BLINK_RIGHT = 'eyeBlinkRight'
    JAW_OPEN = 'jawOpen'
    BROW_DOWN_LEFT = 'browDownLeft'
    BROW_DOWN_RIGHT = 'browDownRight'

    def __init__(self, modelPath='face_landmarker.task', maxFaces=1,
                 minDetectionCon=0.5, minTrackingCon=0.5, outputBlendshapes=True):
        """
        :param modelPath: Path to face_landmarker.task model file
        :param maxFaces: Maximum number of faces to detect
        :param minDetectionCon: Minimum detection confidence
        :param minTrackingCon: Minimum tracking confidence
        :param outputBlendshapes: Whether to compute blendshape scores
        """
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=modelPath),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=maxFaces,
            min_face_detection_confidence=minDetectionCon,
            min_tracking_confidence=minTrackingCon,
            output_face_blendshapes=outputBlendshapes,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.outputBlendshapes = outputBlendshapes

    def findFaces(self, img, draw=True):
        """
        Detect face landmarks in a BGR image.
        :param img: Input BGR image
        :param draw: Draw landmarks on image
        :return: list of face dicts, image
                 Each dict: {"landmarks": [[x,y],...], "blendshapes": {name: score,...}}
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        result = self.landmarker.detect(mp_image)

        faces = []
        h, w, _ = img.shape

        for i, face_landmarks in enumerate(result.face_landmarks):
            landmarks = [[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks]

            faceInfo = {"landmarks": landmarks}

            if self.outputBlendshapes and result.face_blendshapes:
                faceInfo["blendshapes"] = {
                    bs.category_name: round(bs.score, 3)
                    for bs in result.face_blendshapes[i]
                }

            faces.append(faceInfo)

            if draw:
                for pt in landmarks:
                    cv2.circle(img, tuple(pt), 1, (0, 255, 0), cv2.FILLED)

        if draw:
            return faces, img
        return faces

    def getBlendshape(self, faceInfo, name):
        """
        Get a specific blendshape score by name.
        :param faceInfo: Face dict from findFaces()
        :param name: Blendshape name (e.g. FaceLandmarker.SMILE_LEFT)
        :return: Score between 0.0 and 1.0
        """
        return faceInfo.get("blendshapes", {}).get(name, 0.0)

    def isSmiling(self, faceInfo, threshold=0.5):
        """Detect if the face is smiling."""
        left = self.getBlendshape(faceInfo, self.SMILE_LEFT)
        right = self.getBlendshape(faceInfo, self.SMILE_RIGHT)
        return (left + right) / 2 > threshold

    def isBlinking(self, faceInfo, eye='both', threshold=0.4):
        """
        Detect eye blink.
        :param eye: 'left', 'right', or 'both'
        :return: True if blinking
        """
        left = self.getBlendshape(faceInfo, self.BLINK_LEFT) > threshold
        right = self.getBlendshape(faceInfo, self.BLINK_RIGHT) > threshold
        if eye == 'left':
            return left
        if eye == 'right':
            return right
        return left or right

    def isMouthOpen(self, faceInfo, threshold=0.3):
        """Detect if the mouth is open."""
        return self.getBlendshape(faceInfo, self.JAW_OPEN) > threshold

    def getExpression(self, faceInfo, threshold=0.5):
        """
        Return the dominant expression above threshold.
        :return: expression name string or 'neutral'
        """
        blendshapes = faceInfo.get("blendshapes", {})
        expressions = {
            'smiling': (blendshapes.get(self.SMILE_LEFT, 0) + blendshapes.get(self.SMILE_RIGHT, 0)) / 2,
            'mouth_open': blendshapes.get(self.JAW_OPEN, 0),
            'blink_left': blendshapes.get(self.BLINK_LEFT, 0),
            'blink_right': blendshapes.get(self.BLINK_RIGHT, 0),
            'brow_down': (blendshapes.get(self.BROW_DOWN_LEFT, 0) + blendshapes.get(self.BROW_DOWN_RIGHT, 0)) / 2,
        }
        dominant = max(expressions, key=expressions.get)
        return dominant if expressions[dominant] > threshold else 'neutral'


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceLandmarker()
    while True:
        success, img = cap.read()
        faces, img = detector.findFaces(img)
        if faces:
            expr = detector.getExpression(faces[0])
            smiling = detector.isSmiling(faces[0])
            cv2.putText(img, f'Expr: {expr}  Smile: {smiling}',
                        (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("FaceLandmarker", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
