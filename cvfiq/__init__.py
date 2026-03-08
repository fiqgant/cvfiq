import cv2  # re-exported so users never need to import cv2 separately

from cvfiq.Utils import stackImages, cornerRect, findContours, overlayPNG, rotateImage, putTextRect
from cvfiq.ColorModule import ColorFinder
from cvfiq.FPS import FPS
from cvfiq.PIDModule import PID
from cvfiq.PlotModule import LivePlot
from cvfiq.FaceDetectionModule import FaceDetector
from cvfiq.FaceMeshModule import FaceMeshDetector
from cvfiq.HandTrackingModule import HandDetector
from cvfiq.PoseModule import PoseDetector
from cvfiq.SelfiSegmentationModule import SelfiSegmentation
from cvfiq.SerialModule import SerialObject
from cvfiq.ClassificationModule import Classifier
from cvfiq.ArucoModule import ArucoDetector
from cvfiq.DNNModule import DNNClassifier
from cvfiq.VideoStabilizerModule import VideoStabilizer

# MediaPipe Tasks API modules (require model files)
try:
    from cvfiq.GestureModule import GestureDetector
    from cvfiq.FaceLandmarkerModule import FaceLandmarker
    from cvfiq.ObjectDetectorModule import ObjectDetector
except Exception:
    pass


# ── Auto-download helper ──────────────────────────────────────────────────────

_MODEL_URLS = {
    'gesture_recognizer.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task'
    ),
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
    ),
    'efficientdet_lite0.tflite': (
        'https://storage.googleapis.com/mediapipe-models/'
        'object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite'
    ),
}


def _ensure_model(filename):
    """Download model file to current directory if not already present."""
    import os, urllib.request
    if os.path.exists(filename):
        return filename
    url = _MODEL_URLS.get(filename)
    if url is None:
        return filename
    print(f"[cvfiq] Downloading {filename} ...")
    def _progress(count, block, total):
        pct = min(count * block * 100 // total, 100)
        print(f"\r[cvfiq] {filename}: {pct}%", end='', flush=True)
    urllib.request.urlretrieve(url, filename, reporthook=_progress)
    print()
    return filename


# ── Shorthand factory functions ───────────────────────────────────────────────

def hand(**kwargs):
    """Create a HandDetector. Same params as HandDetector()."""
    return HandDetector(**kwargs)

def face(**kwargs):
    """Create a FaceDetector. Same params as FaceDetector()."""
    return FaceDetector(**kwargs)

def mesh(**kwargs):
    """Create a FaceMeshDetector. Same params as FaceMeshDetector()."""
    return FaceMeshDetector(**kwargs)

def pose(**kwargs):
    """Create a PoseDetector. Same params as PoseDetector()."""
    return PoseDetector(**kwargs)

def segment(**kwargs):
    """Create a SelfiSegmentation. Same params as SelfiSegmentation()."""
    return SelfiSegmentation(**kwargs)

def color(**kwargs):
    """Create a ColorFinder. Same params as ColorFinder()."""
    return ColorFinder(**kwargs)

def aruco(**kwargs):
    """Create an ArucoDetector. Same params as ArucoDetector()."""
    return ArucoDetector(**kwargs)

def stabilizer(**kwargs):
    """Create a VideoStabilizer. Same params as VideoStabilizer()."""
    return VideoStabilizer(**kwargs)

def pid(pidVals, targetVal, **kwargs):
    """Create a PID controller."""
    return PID(pidVals, targetVal, **kwargs)

def fps(**kwargs):
    """Create an FPS counter."""
    return FPS(**kwargs)

def plot(**kwargs):
    """Create a LivePlot."""
    return LivePlot(**kwargs)

def classify(modelPath, labelsPath, **kwargs):
    """Create a Classifier (Teachable Machine / Keras model)."""
    return Classifier(modelPath, labelsPath, **kwargs)

def dnn(modelPath, labelsPath, **kwargs):
    """Create a DNNClassifier (ONNX / TFLite via OpenCV DNN)."""
    return DNNClassifier(modelPath, labelsPath, **kwargs)

def serial(**kwargs):
    """Create a SerialObject for Arduino communication."""
    return SerialObject(**kwargs)

def gesture(modelPath='gesture_recognizer.task', **kwargs):
    """Create a GestureDetector. Model auto-downloaded if not present."""
    return GestureDetector(_ensure_model(modelPath), **kwargs)

def landmarker(modelPath='face_landmarker.task', **kwargs):
    """Create a FaceLandmarker. Model auto-downloaded if not present."""
    return FaceLandmarker(_ensure_model(modelPath), **kwargs)

def detector(modelPath='efficientdet_lite0.tflite', **kwargs):
    """Create an ObjectDetector. Model auto-downloaded if not present."""
    return ObjectDetector(_ensure_model(modelPath), **kwargs)
