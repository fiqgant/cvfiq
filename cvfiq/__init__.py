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


# ── Shorthand factory functions ───────────────────────────────────────────────

def hand(**kwargs):
    """Shorthand: create a HandDetector. Same params as HandDetector()."""
    return HandDetector(**kwargs)

def face(**kwargs):
    """Shorthand: create a FaceDetector. Same params as FaceDetector()."""
    return FaceDetector(**kwargs)

def mesh(**kwargs):
    """Shorthand: create a FaceMeshDetector. Same params as FaceMeshDetector()."""
    return FaceMeshDetector(**kwargs)

def pose(**kwargs):
    """Shorthand: create a PoseDetector. Same params as PoseDetector()."""
    return PoseDetector(**kwargs)

def segment(**kwargs):
    """Shorthand: create a SelfiSegmentation. Same params as SelfiSegmentation()."""
    return SelfiSegmentation(**kwargs)

def color(**kwargs):
    """Shorthand: create a ColorFinder. Same params as ColorFinder()."""
    return ColorFinder(**kwargs)

def aruco(**kwargs):
    """Shorthand: create an ArucoDetector. Same params as ArucoDetector()."""
    return ArucoDetector(**kwargs)

def stabilizer(**kwargs):
    """Shorthand: create a VideoStabilizer. Same params as VideoStabilizer()."""
    return VideoStabilizer(**kwargs)

def pid(pidVals, targetVal, **kwargs):
    """Shorthand: create a PID controller."""
    return PID(pidVals, targetVal, **kwargs)

def fps(**kwargs):
    """Shorthand: create an FPS counter."""
    return FPS(**kwargs)

def plot(**kwargs):
    """Shorthand: create a LivePlot."""
    return LivePlot(**kwargs)

def classify(modelPath, labelsPath, **kwargs):
    """Shorthand: create a Classifier."""
    return Classifier(modelPath, labelsPath, **kwargs)

def dnn(modelPath, labelsPath, **kwargs):
    """Shorthand: create a DNNClassifier."""
    return DNNClassifier(modelPath, labelsPath, **kwargs)

def serial(**kwargs):
    """Shorthand: create a SerialObject."""
    return SerialObject(**kwargs)

def gesture(modelPath, **kwargs):
    """Shorthand: create a GestureDetector (requires model file)."""
    return GestureDetector(modelPath, **kwargs)

def landmarker(modelPath, **kwargs):
    """Shorthand: create a FaceLandmarker (requires model file)."""
    return FaceLandmarker(modelPath, **kwargs)

def detector(modelPath, **kwargs):
    """Shorthand: create an ObjectDetector (requires model file)."""
    return ObjectDetector(modelPath, **kwargs)
