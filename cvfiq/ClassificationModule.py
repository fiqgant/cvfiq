"""
Classification Module
Based on Teachable Machine
https://teachablemachine.withgoogle.com/
"""

import numpy as np
import cv2


def _load_keras():
    try:
        import keras
        return keras.models.load_model
    except ImportError:
        pass
    try:
        from tensorflow import keras
        return keras.models.load_model
    except ImportError:
        raise ImportError(
            "ClassificationModule requires keras or tensorflow. "
            "Install with: pip install cvfiq[keras] or pip install cvfiq[tensorflow]"
        )


class Classifier:

    def __init__(self, modelPath, labelsPath=None, imgSize=224):
        """
        :param modelPath: Path to the Keras model (.h5 or SavedModel)
        :param labelsPath: Path to the labels text file
        :param imgSize: Input image size the model expects (default 224)
        """
        self.model_path = modelPath
        self.imgSize = imgSize
        np.set_printoptions(suppress=True)
        _load_model = _load_keras()
        self.model = _load_model(self.model_path)
        self.data = np.ndarray(shape=(1, self.imgSize, self.imgSize, 3), dtype=np.float32)
        self.labels_path = labelsPath
        if self.labels_path:
            with open(self.labels_path, "r") as label_file:
                self.list_labels = [line.strip() for line in label_file]
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0, 255, 0)):
        """
        Run inference on an image.
        :return: (predictions list, index of top class, confidence score of top class)
        """
        imgS = cv2.resize(img, (self.imgSize, self.imgSize))
        image_array = np.asarray(imgS)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        self.data[0] = normalized_image_array

        prediction = self.model.predict(self.data, verbose=0)
        indexVal = int(np.argmax(prediction))
        confidence = float(prediction[0][indexVal])

        if draw and self.labels_path:
            label = self.list_labels[indexVal]
            cv2.putText(img, f'{label} {confidence:.2f}',
                        pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), indexVal, confidence



def main():
    cap = cv2.VideoCapture(0)
    maskClassifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')
    while True:
        _, img = cap.read()
        predection = maskClassifier.getPrediction(img)
        print(predection)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
