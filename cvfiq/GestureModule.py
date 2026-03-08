"""
Gesture Recognition Module
Uses MediaPipe Tasks API GestureRecognizer.
Requires model file: gesture_recognizer.task
Download: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
"""

import cv2
import mediapipe as mp


class GestureDetector:
    """
    Detects hand gestures using MediaPipe Tasks GestureRecognizer.
    Built-in gestures: None, Closed_Fist, Open_Palm, Pointing_Up,
                       Thumb_Down, Thumb_Up, Victory, ILoveYou
    """

    def __init__(self, modelPath='gesture_recognizer.task', numHands=2,
                 minDetectionCon=0.5, minTrackingCon=0.5):
        """
        :param modelPath: Path to gesture_recognizer.task model file
        :param numHands: Maximum number of hands to detect
        :param minDetectionCon: Minimum detection confidence
        :param minTrackingCon: Minimum tracking confidence
        """
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=modelPath),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=numHands,
            min_hand_detection_confidence=minDetectionCon,
            min_tracking_confidence=minTrackingCon,
        )
        self.recognizer = GestureRecognizer.create_from_options(options)

    def findGestures(self, img, draw=True):
        """
        Detect gestures in a BGR image.
        :param img: Input BGR image
        :param draw: Draw landmarks and gesture label on image
        :return: list of gesture dicts, image
                 Each dict: {"gesture", "score", "hand", "lmList"}
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        result = self.recognizer.recognize(mp_image)

        gestures = []
        h, w, _ = img.shape

        if result.gestures:
            for gesture_list, handedness_list, hand_lms in zip(
                    result.gestures, result.handedness, result.hand_landmarks):

                gesture = gesture_list[0]
                handedness = handedness_list[0]

                lmList = [[int(lm.x * w), int(lm.y * h)] for lm in hand_lms]

                gestureInfo = {
                    "gesture": gesture.category_name,
                    "score": round(gesture.score, 3),
                    "hand": "Left" if handedness.category_name == "Right" else "Right",
                    "lmList": lmList,
                }
                gestures.append(gestureInfo)

                if draw:
                    for pt in lmList:
                        cv2.circle(img, tuple(pt), 5, (255, 0, 255), cv2.FILLED)
                    if lmList:
                        x, y = lmList[0]
                        cv2.putText(img,
                                    f'{gestureInfo["gesture"]} ({gestureInfo["score"]:.2f})',
                                    (x - 20, y - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        if draw:
            return gestures, img
        return gestures


def main():
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()
    while True:
        success, img = cap.read()
        gestures, img = detector.findGestures(img)
        for g in gestures:
            print(g["hand"], g["gesture"], g["score"])
        cv2.imshow("Gesture", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
