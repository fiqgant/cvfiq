import cv2

# ── cv2 passthrough — users never need to import cv2 separately ───────────────
from cv2 import (
    VideoCapture, VideoWriter, imread, imwrite, imshow, waitKey,
    destroyAllWindows, destroyWindow, namedWindow, moveWindow,
    resize, flip, rotate, cvtColor, split, merge,
    GaussianBlur, medianBlur, bilateralFilter,
    Canny, dilate, erode, threshold, morphologyEx,
    rectangle, circle, line, ellipse, polylines, fillPoly, putText,
    addWeighted, bitwise_and, bitwise_or, bitwise_not,
    getTextSize, getRotationMatrix2D, warpAffine, applyColorMap,
    COLORMAP_MAGMA, COLORMAP_JET, COLORMAP_PLASMA, COLORMAP_VIRIDIS,
    FONT_HERSHEY_PLAIN, FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_DUPLEX,
    COLOR_BGR2RGB, COLOR_BGR2GRAY, COLOR_BGR2HSV, COLOR_GRAY2BGR,
    FILLED, LINE_AA, LINE_8,
    CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
    INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_NEAREST,
    ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180,
    MORPH_CLOSE, MORPH_OPEN, MORPH_ELLIPSE, MORPH_RECT,
    NORM_MINMAX, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE,
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
from cvfiq.ArucoModule import ArucoDetector
from cvfiq.DNNModule import DNNClassifier
from cvfiq.VideoStabilizerModule import VideoStabilizer
from cvfiq.QRModule import QRDetector
from cvfiq.MotionModule import MotionDetector
from cvfiq.TrackerModule import ObjectTracker

# MediaPipe Tasks API modules (require model files, auto-downloaded)
try:
    from cvfiq.GestureModule import GestureDetector
    from cvfiq.FaceLandmarkerModule import FaceLandmarker
    from cvfiq.ObjectDetectorModule import ObjectDetector
except Exception:
    pass

# Optional heavy modules — graceful fallback if deps not installed
try:
    from cvfiq.ClassificationModule import Classifier
except Exception:
    pass

try:
    from cvfiq.OCRModule import OCRReader
except Exception:
    pass

try:
    from cvfiq.DepthModule import DepthEstimator
except Exception:
    pass

try:
    from cvfiq.EmotionModule import EmotionDetector
except Exception:
    pass

try:
    from cvfiq.AgeGenderModule import AgeGenderDetector
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

    def __init__(self, source=0, title="cvfiq", showFPS=False, quitKey='q',
                 width=None, height=None):
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
        """Display frame. Returns key pressed. Stops iteration on quit key."""
        if self._fpsCounter is not None:
            h = img.shape[0]
            _, img = self._fpsCounter.update(img, pos=(10, h - 10), scale=1,
                                             color=(0, 255, 0), thickness=2)
        cv2.imshow(title or self.title, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(self.quitKey):
            self._running = False
        return key

    def __exit__(self, *args):
        self.cap.release()
        cv2.destroyAllWindows()


# ── run() — simplest possible camera app ─────────────────────────────────────

def run(fn, source=0, title="cvfiq", showFPS=True, quitKey='q',
        width=None, height=None, show=True, record=None, keys=None):
    """
    Open camera/video, call fn(img) every frame, display the result.

    Parameters:
        fn      : function(img) -> img   — your processing function
        source  : int or str  — camera index (0,1,…) or video file path
        title   : window title
        showFPS : overlay FPS counter (default True)
        quitKey : key to exit (default 'q')
        width/height : set capture resolution
        show    : display window (False = headless/server mode)
        record  : file path to save output video, e.g. "output.mp4"
        keys    : dict of extra key handlers, e.g. {'s': lambda img: cvfiq.snapshot(img)}

    Usage:
        import cvfiq
        hand = cvfiq.hand()

        def process(img):
            hands, img = hand.findHands(img)
            return img

        cvfiq.run(process)
        cvfiq.run(process, source=1)               # second camera
        cvfiq.run(process, source="video.mp4")     # video file
        cvfiq.run(process, show=False, record="out.mp4")  # headless + save
        cvfiq.run(process, keys={'s': lambda img: cvfiq.snapshot(img)})
    """
    cap = cv2.VideoCapture(source)
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    writer     = None
    fpsCounter = FPS() if (showFPS and show) else None

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            out = fn(img)
            if out is None:
                out = img

            if record:
                if writer is None:
                    h, w  = out.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(record, fourcc, 30, (w, h))
                writer.write(out)

            if show:
                display = out.copy()
                if fpsCounter:
                    h = display.shape[0]
                    _, display = fpsCounter.update(display, pos=(10, h - 10), scale=1,
                                                   color=(0, 255, 0), thickness=2)
                cv2.imshow(title, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(quitKey):
                    break
                if keys:
                    handler = keys.get(chr(key))
                    if handler:
                        handler(out)
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        if writer:
            writer.release()


# ── pipeline() — chain multiple process functions ────────────────────────────

def pipeline(*fns):
    """
    Chain multiple process functions into one.
    Each function receives img and must return img.

    Usage:
        hand = cvfiq.hand()
        face = cvfiq.face()

        pipe = cvfiq.pipeline(
            lambda img: hand.findHands(img)[1],
            lambda img: face.findFaces(img)[0],
        )

        cvfiq.run(pipe)
    """
    def _run(img):
        for fn in fns:
            result = fn(img)
            img = result if result is not None else img
        return img
    return _run


# ── text() — one-line putText shorthand ──────────────────────────────────────

def text(img, txt, pos, color=(255, 255, 255), scale=1, thickness=2, bg=False):
    """
    Draw text on img. Shorthand for cv2.putText / cvfiq.putTextRect.

    Usage:
        cvfiq.text(img, "Hello", (10, 30))
        cvfiq.text(img, f"FPS: {f:.0f}", (10, 30), color=(0,255,0))
        cvfiq.text(img, "Label", (10, 50), bg=True)   # with background rect
    """
    if bg:
        img, _ = putTextRect(img, str(txt), pos, scale=scale, thickness=thickness)
    else:
        cv2.putText(img, str(txt), pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return img


# ── snapshot() — save frame to file ──────────────────────────────────────────

def snapshot(img, prefix="snap", folder="."):
    """
    Save img as a numbered JPEG file.

    Usage:
        cvfiq.run(process, keys={'s': lambda img: cvfiq.snapshot(img)})
        # creates snap_001.jpg, snap_002.jpg, ...
    """
    import os, glob
    existing = glob.glob(os.path.join(folder, f"{prefix}_*.jpg"))
    n    = len(existing) + 1
    path = os.path.join(folder, f"{prefix}_{n:03d}.jpg")
    cv2.imwrite(path, img)
    print(f"[cvfiq] Saved: {path}")
    return path


# ── Shorthand factory functions ───────────────────────────────────────────────

def hand(**kwargs):
    """Create a HandDetector."""
    return HandDetector(**kwargs)

def face(**kwargs):
    """Create a FaceDetector."""
    return FaceDetector(**kwargs)

def mesh(**kwargs):
    """Create a FaceMeshDetector."""
    return FaceMeshDetector(**kwargs)

def pose(**kwargs):
    """Create a PoseDetector."""
    return PoseDetector(**kwargs)

def segment(**kwargs):
    """Create a SelfiSegmentation."""
    return SelfiSegmentation(**kwargs)

def color(**kwargs):
    """Create a ColorFinder."""
    return ColorFinder(**kwargs)

def aruco(**kwargs):
    """Create an ArucoDetector."""
    return ArucoDetector(**kwargs)

def stabilizer(**kwargs):
    """Create a VideoStabilizer."""
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

def qr(**kwargs):
    """Create a QRDetector for QR codes and barcodes."""
    return QRDetector(**kwargs)

def motion(**kwargs):
    """Create a MotionDetector using background subtraction."""
    return MotionDetector(**kwargs)

def tracker(algo='CSRT', **kwargs):
    """Create an ObjectTracker. algo: CSRT (default), KCF, MIL, MOSSE."""
    return ObjectTracker(algo=algo, **kwargs)

def ocr(**kwargs):
    """Create an OCRReader. Requires: pip install pytesseract + tesseract."""
    return OCRReader(**kwargs)

def depth(**kwargs):
    """Create a DepthEstimator (MiDaS). Model auto-downloaded on first use."""
    return DepthEstimator(**kwargs)

def emotion(**kwargs):
    """Create an EmotionDetector. Requires: pip install deepface."""
    return EmotionDetector(**kwargs)

def age_gender(**kwargs):
    """Create an AgeGenderDetector. Models auto-downloaded on first use."""
    return AgeGenderDetector(**kwargs)
