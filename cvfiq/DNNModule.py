"""
DNN Classification Module
Uses OpenCV DNN module — no TensorFlow/Keras needed.
Supports ONNX, TFLite, Caffe, and other formats readable by cv2.dnn.
Lighter alternative to ClassificationModule.
"""

import cv2
import numpy as np


class DNNClassifier:
    """
    Image classifier using OpenCV's built-in DNN module.
    Supports ONNX, TFLite, Caffe models — no TensorFlow required.
    """

    def __init__(self, modelPath, labelsPath=None, imgSize=(224, 224),
                 mean=(127.5, 127.5, 127.5), scale=1 / 127.5, swapRB=True):
        """
        :param modelPath: Path to model file (.onnx, .tflite, .caffemodel, etc.)
        :param labelsPath: Path to labels text file (one label per line)
        :param imgSize: Input size expected by the model (width, height)
        :param mean: Mean subtraction values for preprocessing
        :param scale: Scale factor for preprocessing
        :param swapRB: Swap Red and Blue channels (True for BGR→RGB)
        """
        self.net = cv2.dnn.readNet(modelPath)
        self.imgSize = imgSize
        self.mean = mean
        self.scale = scale
        self.swapRB = swapRB
        self.list_labels = []

        if labelsPath:
            with open(labelsPath, 'r') as f:
                self.list_labels = [line.strip() for line in f]

        # Use GPU if available
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def useGPU(self):
        """Enable CUDA GPU acceleration if available. Returns True if successful."""
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            return True
        except cv2.error:
            import logging
            logging.warning("CUDA not available. Falling back to CPU.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return False

    def getPrediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0, 255, 0)):
        """
        Run inference on an image.
        :param img: Input BGR image
        :param draw: Draw prediction label on image
        :return: (predictions list, index of top class, confidence score)
        """
        blob = cv2.dnn.blobFromImage(
            img, self.scale, self.imgSize, self.mean, swapRB=self.swapRB)
        self.net.setInput(blob)
        output = self.net.forward()

        scores = output[0]
        indexVal = int(np.argmax(scores))
        confidence = float(scores[indexVal])

        if draw and self.list_labels:
            label = self.list_labels[indexVal] if indexVal < len(self.list_labels) else str(indexVal)
            cv2.putText(img, f'{label} {confidence:.2f}',
                        pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(scores), indexVal, confidence

    def getTopK(self, img, k=3):
        """
        Return top-K predictions with labels and scores.
        :param img: Input BGR image
        :param k: Number of top results
        :return: list of (label, score) sorted by score descending
        """
        blob = cv2.dnn.blobFromImage(
            img, self.scale, self.imgSize, self.mean, swapRB=self.swapRB)
        self.net.setInput(blob)
        output = self.net.forward()[0]

        top_indices = np.argsort(output)[::-1][:k]
        results = []
        for idx in top_indices:
            label = self.list_labels[idx] if idx < len(self.list_labels) else str(idx)
            results.append((label, round(float(output[idx]), 3)))
        return results


def main():
    cap = cv2.VideoCapture(0)
    # Replace with your ONNX or TFLite model
    classifier = DNNClassifier('model.onnx', 'labels.txt')
    while True:
        success, img = cap.read()
        scores, idx, conf = classifier.getPrediction(img)
        cv2.imshow("DNN", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
