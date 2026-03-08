import cv2

# ── cv2 passthrough — users never need to import cv2 separately ───────────────
from cv2 import (
    VideoCapture, VideoWriter, imread, imwrite, imshow, waitKey,
    destroyAllWindows, destroyWindow, namedWindow, moveWindow,
    resize, flip, rotate, cvtColor, split, merge,
    GaussianBlur, medianBlur, bilateralFilter,
    Canny, dilate, erode, threshold,
    rectangle, circle, line, ellipse, polylines, fillPoly, putText,
    addWeighted, bitwise_and, bitwise_or, bitwise_not,
    getTextSize, getRotationMatrix2D, warpAffine,
    FONT_HERSHEY_PLAIN, FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_DUPLEX,
    COLOR_BGR2RGB, COLOR_BGR2GRAY, COLOR_BGR2HSV, COLOR_GRAY2BGR,
    FILLED, LINE_AA, LINE_8,
    CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
    INTER_LINEAR, INTER_AREA, INTER_CUBIC,
    ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180,
)

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

# MediaPipe Tasks API modules (require model files, auto-downloaded)
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


# ── Camera — replaces the 7-line capture loop boilerplate ─────────────────────

class Camera:
    """
    Context manager that wraps VideoCapture.
    Iterate to get frames; call cam.show(img) to display and handle quit.

    Usage:
        with cvfiq.Camera(0, showFPS=True) as cam:
            hand = cvfiq.hand()
            for img in cam:
                hands, img = hand.findHands(img)
                cam.show(img)
    """

    def __init__(self, source=0, title="cvfiq", showFPS=False, quitKey='q', width=None, height=None):
        self.cap      = cv2.VideoCapture(source)
        self.title    = title
        self.showFPS  = showFPS
        self.quitKey  = quitKey
        self._running = False
        self._fpsCounter = FPS() if showFPS else None

        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __enter__(self):
        self._running = True
        return self

    def __iter__(self):
        while self._running:
            success, img = self.cap.read()
            if not success:
                break
            yield img

    def show(self, img, title=None):
        """Display frame. Stops iteration when quit key is pressed."""
        if self._fpsCounter is not None:
            _, img = self._fpsCounter.update(img)
        cv2.imshow(title or self.title, img)
        if cv2.waitKey(1) & 0xFF == ord(self.quitKey):
            self._running = False

    def __exit__(self, *args):
        self.cap.release()
        cv2.destroyAllWindows()


# ── run() — simplest possible camera app in 4 lines ──────────────────────────

def run(fn, source=0, title="cvfiq", showFPS=True, quitKey='q',
        width=None, height=None):
    """
    Open camera, call fn(img) every frame, display the result.
    Press quitKey (default 'q') to exit.

    Usage:
        import cvfiq
        hand = cvfiq.hand()

        def process(img):
            hands, img = hand.findHands(img)
            return img

        cvfiq.run(process)
    """
    with Camera(source, title=title, showFPS=showFPS, quitKey=quitKey,
                width=width, height=height) as cam:
        for img in cam:
            result = fn(img)
            cam.show(result if result is not None else img)


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
