"""
Object Detection Module
Uses MediaPipe Tasks API ObjectDetector.
Detects common objects (80 COCO classes).
Requires model file: efficientdet_lite0.tflite
Download: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite
"""

import cv2
import mediapipe as mp


class ObjectDetector:
    """
    Detects common objects in real-time using MediaPipe Tasks ObjectDetector.
    Supports 80 COCO object classes (person, car, bottle, chair, etc.)
    """

    def __init__(self, modelPath='efficientdet_lite0.tflite',
                 maxResults=5, scoreThreshold=0.5):
        """
        :param modelPath: Path to .tflite model file
        :param maxResults: Maximum number of detections to return
        :param scoreThreshold: Minimum confidence score to report
        """
        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=modelPath),
            running_mode=VisionRunningMode.IMAGE,
            max_results=maxResults,
            score_threshold=scoreThreshold,
        )
        self.detector = ObjectDetector.create_from_options(options)

    def findObjects(self, img, draw=True):
        """
        Detect objects in a BGR image.
        :param img: Input BGR image
        :param draw: Draw bounding boxes and labels on image
        :return: list of object dicts, image
                 Each dict: {"label", "score", "bbox": (x,y,w,h), "center": (cx,cy)}
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        result = self.detector.detect(mp_image)

        objects = []

        for detection in result.detections:
            bbox = detection.bounding_box
            x, y = bbox.origin_x, bbox.origin_y
            bw, bh = bbox.width, bbox.height
            cx, cy = x + bw // 2, y + bh // 2

            category = detection.categories[0]
            objInfo = {
                "label": category.category_name,
                "score": round(category.score, 3),
                "bbox": (x, y, bw, bh),
                "center": (cx, cy),
            }
            objects.append(objInfo)

            if draw:
                cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(img,
                            f'{category.category_name} {category.score:.2f}',
                            (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        if draw:
            return objects, img
        return objects

    def getObjectsByLabel(self, objects, label):
        """
        Filter detected objects by label name.
        :param objects: List from findObjects()
        :param label: Label string to filter (e.g. 'person', 'car')
        :return: Filtered list
        """
        return [o for o in objects if o["label"].lower() == label.lower()]

    def countObjects(self, objects, label=None):
        """
        Count detected objects, optionally filtered by label.
        :param objects: List from findObjects()
        :param label: Optional label filter
        :return: Count
        """
        if label:
            return len(self.getObjectsByLabel(objects, label))
        return len(objects)


def main():
    cap = cv2.VideoCapture(0)
    detector = ObjectDetector()
    while True:
        success, img = cap.read()
        objects, img = detector.findObjects(img)
        cv2.putText(img, f'Objects: {len(objects)}',
                    (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        cv2.imshow("ObjectDetector", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
