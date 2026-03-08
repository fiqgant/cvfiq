# cvfiq

A Python Computer Vision library that makes it easy to run image processing and AI functions using OpenCV and MediaPipe.

## Requirements

| Library | Min | Max |
|---|---|---|
| Python | 3.8 | 3.12 |
| opencv-python | 4.7.0 | 4.13.0.92 |
| mediapipe | 0.10.0 | 0.10.32 |
| numpy | 1.21.0 | 2.4.2 |
| pyserial | 3.0 | 3.5 |

## Installation

```bash
pip install cvfiq
```

For `ClassificationModule` (optional, choose one):
```bash
pip install cvfiq[keras]        # standalone Keras
pip install cvfiq[tensorflow]   # TensorFlow
```

---

## Modules

### Face Detection

```python
from cvfiq.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)

    if bboxs:
        # bboxInfo keys: "id", "bbox", "score", "center", "keypoints"
        center = bboxs[0]["center"]
        keypoints = bboxs[0]["keypoints"]  # rightEye, leftEye, nose, mouth, rightEar, leftEar
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### Hand Tracking

```python
from cvfiq.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]   # 21 landmarks [x, y, z]
        bbox1   = hand1["bbox"]     # x, y, w, h
        center1 = hand1["center"]   # cx, cy
        type1   = hand1["type"]     # "Left" or "Right"

        fingers1 = detector.fingersUp(hand1)  # [thumb, index, middle, ring, pinky]

        # 2D distance between two landmarks
        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[4][0:2], img)

        # 3D distance (uses z-depth)
        length3d = detector.findDistance3D(lmList1[8], lmList1[4])

        # Angle between 3 landmarks (index tip - wrist - thumb tip)
        angle = detector.findAngle(lmList1[8], lmList1[0], lmList1[4])

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### Pose Estimation

```python
from cvfiq.PoseModule import PoseDetector
import cv2

cap = cv2.VideoCapture(0)
detector = PoseDetector(smoothAlpha=0.5)  # 0=max smooth, 1=no smooth

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)

    if lmList:
        # Calculate angle at elbow (landmarks 11, 13, 15)
        angle = detector.findAngle(img, 11, 13, 15)
        # Check if angle is within range
        isCorrect = detector.angleCheck(angle, targetAngle=90, addOn=20)

    if bboxInfo:
        center = bboxInfo["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### Face Mesh

```python
from cvfiq.FaceMeshModule import FaceMeshDetector
import cv2

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=2, refineLandmarks=False)

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img)

    if faces:
        face = faces[0]

        # Get specific region landmarks
        leftEye  = detector.getRegion(face, FaceMeshDetector.LEFT_EYE)
        lips     = detector.getRegion(face, FaceMeshDetector.LIPS)

        # Detect blink
        blink = detector.blinkDetector(face)
        # {"left": bool, "right": bool, "leftEAR": float, "rightEAR": float}

        # Detect mouth open
        mouth = detector.mouthOpen(face)
        # {"open": bool, "ratio": float}

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### Selfie Segmentation (Background Removal)

```python
from cvfiq.SelfiSegmentationModule import SelfiSegmentation
import cv2

cap = cv2.VideoCapture(0)
segmentor = SelfiSegmentation(model=1)  # 0=general, 1=landscape (faster)
imgBg = cv2.imread("background.jpg")

while True:
    success, img = cap.read()

    # Image background with smooth edges
    imgOut = segmentor.removeBG(img, imgBg=imgBg, threshold=0.1, smooth=True, kernelSize=11)

    # Solid color background
    # imgOut = segmentor.removeBG(img, imgBg=(0, 255, 0), smooth=True)

    # Blur background
    # blurred = cv2.GaussianBlur(img, (55, 55), 0)
    # imgOut = segmentor.removeBG(img, imgBg=blurred, smooth=True)

    cv2.imshow("Output", imgOut)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### Gesture Recognition *(requires model file)*

```python
from cvfiq.GestureModule import GestureDetector
import cv2

# Download model: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
cap = cv2.VideoCapture(0)
detector = GestureDetector(modelPath='gesture_recognizer.task')

while True:
    success, img = cap.read()
    gestures, img = detector.findGestures(img)

    for g in gestures:
        print(g["hand"], g["gesture"], g["score"])
        # Gestures: Thumb_Up, Thumb_Down, Victory, Open_Palm,
        #           Closed_Fist, Pointing_Up, ILoveYou, None

    cv2.imshow("Gesture", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### Face Landmarker + Expressions *(requires model file)*

```python
from cvfiq.FaceLandmarkerModule import FaceLandmarker
import cv2

# Download model: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
cap = cv2.VideoCapture(0)
detector = FaceLandmarker(modelPath='face_landmarker.task')

while True:
    success, img = cap.read()
    faces, img = detector.findFaces(img)

    if faces:
        face = faces[0]

        smiling    = detector.isSmiling(face)
        blinking   = detector.isBlinking(face, eye='both')
        mouthOpen  = detector.isMouthOpen(face)
        expression = detector.getExpression(face)
        # expressions: smiling, mouth_open, blink_left, blink_right, brow_down, neutral

        # Raw blendshape value (0.0 - 1.0)
        jawScore = detector.getBlendshape(face, FaceLandmarker.JAW_OPEN)

    cv2.imshow("FaceLandmarker", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### Object Detection *(requires model file)*

```python
from cvfiq.ObjectDetectorModule import ObjectDetector
import cv2

# Download model: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite
cap = cv2.VideoCapture(0)
detector = ObjectDetector(modelPath='efficientdet_lite0.tflite', scoreThreshold=0.5)

while True:
    success, img = cap.read()
    objects, img = detector.findObjects(img)

    # Filter by label
    people = detector.getObjectsByLabel(objects, 'person')
    count  = detector.countObjects(objects, 'person')
    # objInfo keys: "label", "score", "bbox": (x,y,w,h), "center": (cx,cy)

    cv2.imshow("ObjectDetector", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### ArUco Marker Detection

```python
from cvfiq.ArucoModule import ArucoDetector
import cv2

cap = cv2.VideoCapture(0)
detector = ArucoDetector(dictType='4x4_50')

while True:
    success, img = cap.read()
    markers, img = detector.findMarkers(img)
    # markerInfo keys: "id", "corners": [[x,y]x4], "center": (cx,cy)

    if markers:
        # Estimate 3D pose (distance in meters)
        poses, img = detector.estimatePose(img, markers, markerSize=0.05)
        for pose in poses:
            print(f"ID:{pose['id']}  dist:{pose['distance']:.3f}m")

    cv2.imshow("ArUco", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# Generate a marker image and save it
detector = ArucoDetector()
markerImg = detector.generateMarker(markerId=0, size=200)
cv2.imwrite("marker_0.png", markerImg)
```

---

### DNN Classification (no TensorFlow needed)

```python
from cvfiq.DNNModule import DNNClassifier
import cv2

cap = cv2.VideoCapture(0)
classifier = DNNClassifier('model.onnx', labelsPath='labels.txt', imgSize=(224, 224))
# classifier.useGPU()  # enable CUDA if available

while True:
    success, img = cap.read()

    # Single prediction
    scores, index, confidence = classifier.getPrediction(img)

    # Top-3 predictions
    topK = classifier.getTopK(img, k=3)
    # [("cat", 0.92), ("dog", 0.05), ("bird", 0.01)]

    cv2.imshow("DNN", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### Video Stabilizer

```python
from cvfiq.VideoStabilizerModule import VideoStabilizer
import cv2

cap = cv2.VideoCapture(0)
stabilizer = VideoStabilizer(smoothRadius=15, border='black')
# border options: 'black', 'replicate', 'reflect'

while True:
    success, img = cap.read()
    imgStab = stabilizer.stabilize(img)

    smoothness = stabilizer.getSmoothness()  # 0.0 (shaky) to 1.0 (smooth)

    cv2.imshow("Stabilized", imgStab)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stabilizer.reset()  # reset when switching video source
cap.release()
cv2.destroyAllWindows()
```

---

### Color Detection

```python
from cvfiq.ColorModule import ColorFinder
import cv2

cap = cv2.VideoCapture(0)
myColorFinder = ColorFinder(trackBar=False)

# Custom color HSV range
hsvVals = {'hmin': 10, 'smin': 55, 'vmin': 215, 'hmax': 42, 'smax': 255, 'vmax': 255}

while True:
    success, img = cap.read()

    # Single color
    imgColor, mask = myColorFinder.update(img, 'red')   # built-in: red, green, blue
    imgColor, mask = myColorFinder.update(img, hsvVals)  # custom HSV

    # Multiple colors at once
    results = myColorFinder.updateMulti(img, ['red', 'green', 'blue'])
    # results["red"]["imgColor"], results["red"]["mask"]

    # Save / load custom color profiles
    myColorFinder.saveColor('orange', hsvVals, 'colors.json')
    loaded = myColorFinder.loadColor('orange', 'colors.json')

    cv2.imshow("Color", imgColor)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### FPS Counter

```python
import cvfiq
import cv2

fpsReader = cvfiq.FPS(avgCount=10)  # average over 10 frames
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    fps, img = fpsReader.update(img, pos=(20, 50), color=(0, 255, 0), scale=3, thickness=3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### PID Controller

```python
from cvfiq.PIDModule import PID
from cvfiq.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)
detector = FaceDetector()

# [Kp, Ki, Kd], target, iLimit = integral windup clamp
xPID = PID([1, 0.0001, 0.5], targetVal=320, iLimit=[-500, 500])
yPID = PID([1, 0.0001, 0.5], targetVal=240, axis=1, limit=[-100, 100], iLimit=[-500, 500])

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)

    if bboxs:
        cx, cy = bboxs[0]["center"]
        xVal = int(xPID.update(cx))
        yVal = int(yPID.update(cy))
        xPID.draw(img, [cx, cy])
        yPID.draw(img, [cx, cy])
    else:
        xPID.reset()  # reset state when face lost
        yPID.reset()

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### Live Plot

```python
import cvfiq
import cv2
import math

xPlot = cvfiq.LivePlot(w=640, h=480, yLimit=[-100, 100])
x = 0

while True:
    x = (x + 1) % 360
    val = int(math.sin(math.radians(x)) * 100)
    imgPlot = xPlot.update(val, color=(255, 0, 255))
    cv2.imshow("Plot", imgPlot)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
```

---

### Stack Images

```python
import cvfiq
import cv2

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgList = [img, img, imgGray, img, imgGray, img]
    stackedImg = cvfiq.stackImages(imgList, cols=3, scale=0.4)
    cv2.imshow("Stacked", stackedImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

### Arduino Serial Communication

```python
from cvfiq.SerialModule import SerialObject
import time

# Auto-detect Arduino, or specify port
arduino = SerialObject(portNo=None, baudRate=9600, digits=3, timeout=1)

while True:
    # Send data: "$001001000" (3 digits per value)
    arduino.sendData([1, 1, 0])

    # Receive data (with timeout — won't hang)
    data = arduino.getData()
    print(data)

    time.sleep(0.1)
```

---

### Classification (Teachable Machine)

```python
from cvfiq.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture(0)
classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt', imgSize=224)

while True:
    success, img = cap.read()

    # Returns: (all scores list, top index, confidence)
    scores, index, confidence = classifier.getPrediction(img)
    print(f"Predicted: index={index}  confidence={confidence:.2f}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

## Utility Functions

```python
import cvfiq
import cv2

img = cv2.imread("image.jpg")

# Stack multiple images into one window
stacked = cvfiq.stackImages([img, img, img], cols=3, scale=0.5)

# Fancy bounding box with corner lines
img = cvfiq.cornerRect(img, bbox=(100, 100, 200, 150))

# Find contours with area filter
imgCon, contours = cvfiq.findContours(img, imgPre, minArea=1000, filter=4)  # filter=4: rectangle only

# Overlay transparent PNG
img = cvfiq.overlayPNG(imgBack=img, imgFront=imgPNG, pos=[50, 50])

# Rotate image
img = cvfiq.rotateImage(img, angle=45, scale=1)

# Text with background rectangle
img, rect = cvfiq.putTextRect(img, "Hello", pos=(50, 50), scale=2, thickness=2)
```

---

## Module Overview

| Module | Class | Description |
|---|---|---|
| `FaceDetectionModule` | `FaceDetector` | Face detection + 6 keypoints |
| `FaceMeshModule` | `FaceMeshDetector` | 468 face landmarks, blink & mouth detection |
| `HandTrackingModule` | `HandDetector` | 21 hand landmarks, finger detection, 2D/3D distance, angle |
| `PoseModule` | `PoseDetector` | Body pose estimation, angle, smoothing |
| `SelfiSegmentationModule` | `SelfiSegmentation` | Real-time background removal |
| `GestureModule` | `GestureDetector` | Built-in gesture recognition (Tasks API) |
| `FaceLandmarkerModule` | `FaceLandmarker` | 478 landmarks + 52 expression blendshapes (Tasks API) |
| `ObjectDetectorModule` | `ObjectDetector` | 80-class object detection (Tasks API) |
| `ArucoModule` | `ArucoDetector` | ArUco marker detection + 3D pose estimation |
| `DNNModule` | `DNNClassifier` | ONNX/TFLite classification via OpenCV DNN |
| `VideoStabilizerModule` | `VideoStabilizer` | Real-time video stabilization |
| `ColorModule` | `ColorFinder` | HSV color detection, save/load profiles, multi-color |
| `ClassificationModule` | `Classifier` | Teachable Machine model inference |
| `FPS` | `FPS` | Averaged FPS counter |
| `PIDModule` | `PID` | PID controller with windup clamp |
| `PlotModule` | `LivePlot` | Real-time graph in OpenCV window |
| `SerialModule` | `SerialObject` | Arduino serial communication |
| `Utils` | — | `stackImages`, `cornerRect`, `findContours`, `overlayPNG`, `rotateImage`, `putTextRect` |

---

## Tests

A full test suite is included in the `tests/` folder.

```bash
cd tests

# Batch test — no camera, no GUI needed
python run_all.py

# Individual module tests (webcam required)
python test_hand_tracking.py
python test_face_detection.py
python test_face_mesh.py
python test_pose.py
python test_selfie_segmentation.py
python test_aruco.py
python test_video_stabilizer.py
python test_color.py

# Tasks API tests — model auto-downloaded if not present
python test_gesture.py
python test_face_landmarker.py
python test_object_detector.py

# GUI tests (no camera)
python test_plot.py
python test_utils.py
```

---

## License

MIT License — see [LICENSE](LICENSE)
