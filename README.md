# cvfiq

A Python Computer Vision library that makes it easy to run image processing and AI functions using OpenCV and MediaPipe.

> **One import** — `cv2` is bundled, no separate install or import needed.
> **Model files are auto-downloaded** — Tasks API detectors work out of the box.
> **`cvfiq.run()`** — handles the camera loop, display, and quit key for you.

## Quick Usage

```python
import cvfiq

hand = cvfiq.hand(maxHands=1)
face = cvfiq.face()

def process(img):
    hands, img = hand.findHands(img)
    img, bboxs = face.findFaces(img)
    return img

# source=0 → default webcam, source=1 → second camera, source="video.mp4" → file
cvfiq.run(process)              # webcam 0, FPS shown, press q to quit
cvfiq.run(process, source=1)    # second camera
cvfiq.run(process, source="video.mp4", showFPS=False)  # video file
```

## Camera Loop (manual control)

```python
import cvfiq

hand = cvfiq.hand(maxHands=1)

# cvfiq.Camera(source, showFPS, title, quitKey, width, height)
with cvfiq.Camera(0, showFPS=True, title="Hand Demo") as cam:
    for img in cam:
        hands, img = hand.findHands(img)
        cam.show(img)
```

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

For optional heavy modules:
```bash
pip install pytesseract          # OCRModule (+ install tesseract binary)
pip install deepface             # EmotionModule
```

---

## Modules

### Face Detection

```python
import cvfiq

detector = cvfiq.face()

def process(img):
    img, bboxs = detector.findFaces(img)
    if bboxs:
        # bboxInfo keys: "id", "bbox", "score", "center", "keypoints"
        cvfiq.circle(img, bboxs[0]["center"], 5, (255, 0, 255), cvfiq.FILLED)
    return img

cvfiq.run(process)
```

---

### Hand Tracking

```python
import cvfiq

detector = cvfiq.hand(detectionCon=0.8, maxHands=2)

def process(img):
    hands, img = detector.findHands(img)
    if hands:
        hand1   = hands[0]
        lmList1 = hand1["lmList"]   # 21 landmarks [x, y, z]
        bbox1   = hand1["bbox"]     # x, y, w, h
        type1   = hand1["type"]     # "Left" or "Right"

        fingers = detector.fingersUp(hand1)            # [thumb, index, middle, ring, pinky]
        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[4][0:2], img)
        length3d = detector.findDistance3D(lmList1[8], lmList1[4])
        angle    = detector.findAngle(lmList1[8], lmList1[0], lmList1[4])
    return img

cvfiq.run(process)
```

---

### Pose Estimation

```python
import cvfiq

detector = cvfiq.pose(smoothAlpha=0.5)  # 0=max smooth, 1=no smooth

def process(img):
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)
    if lmList:
        angle     = detector.findAngle(img, 11, 13, 15)  # shoulder-elbow-wrist
        isCorrect = detector.angleCheck(angle, targetAngle=90, addOn=20)
    return img

cvfiq.run(process)
```

---

### Face Mesh

```python
import cvfiq

detector = cvfiq.mesh(maxFaces=2)

def process(img):
    img, faces = detector.findFaceMesh(img)
    if faces:
        blink = detector.blinkDetector(faces[0])
        # {"left": bool, "right": bool, "leftEAR": float, "rightEAR": float}
        mouth = detector.mouthOpen(faces[0])
        # {"open": bool, "ratio": float}
    return img

cvfiq.run(process)
```

---

### Selfie Segmentation (Background Removal)

```python
import cvfiq

seg   = cvfiq.segment(model=1)  # 0=general, 1=landscape (faster)
imgBg = cvfiq.imread("background.jpg")  # or use a color tuple: (0, 255, 0)

def process(img):
    return seg.removeBG(img, imgBg=imgBg, smooth=True)

cvfiq.run(process)
```

---

### Gesture Recognition *(model auto-downloaded)*

```python
import cvfiq

detector = cvfiq.gesture()  # downloads model automatically on first run

def process(img):
    gestures, img = detector.findGestures(img)
    for g in gestures:
        print(g["hand"], g["gesture"], g["score"])
        # Gestures: Thumb_Up, Thumb_Down, Victory, Open_Palm,
        #           Closed_Fist, Pointing_Up, ILoveYou, None
    return img

cvfiq.run(process)
```

---

### Face Landmarker + Expressions *(model auto-downloaded)*

```python
import cvfiq
from cvfiq.FaceLandmarkerModule import FaceLandmarker  # for blendshape constants

detector = cvfiq.landmarker()  # downloads model automatically on first run

def process(img):
    faces, img = detector.findFaces(img)
    if faces:
        print(detector.getExpression(faces[0]))
        # expressions: smiling, mouth_open, blink_left, blink_right, brow_down, neutral
        jaw = detector.getBlendshape(faces[0], FaceLandmarker.JAW_OPEN)
    return img

cvfiq.run(process)
```

---

### Object Detection *(model auto-downloaded)*

```python
import cvfiq

detector = cvfiq.detector(scoreThreshold=0.5)  # downloads model automatically on first run

def process(img):
    objects, img = detector.findObjects(img)
    count = detector.countObjects(objects, 'person')
    # objInfo keys: "label", "score", "bbox": (x,y,w,h), "center": (cx,cy)
    return img

cvfiq.run(process)
```

---

### ArUco Marker Detection

```python
import cvfiq

detector = cvfiq.aruco(dictType='4x4_50')

def process(img):
    markers, img = detector.findMarkers(img)
    # markerInfo keys: "id", "corners": [[x,y]x4], "center": (cx,cy)
    if markers:
        poses, img = detector.estimatePose(img, markers, markerSize=0.05)
        for pose in poses:
            print(f"ID:{pose['id']}  dist:{pose['distance']:.3f}m")
    return img

cvfiq.run(process)

# Generate and save a marker image (no camera needed)
markerImg = cvfiq.aruco().generateMarker(markerId=0, size=200)
cvfiq.imwrite("marker_0.png", markerImg)
```

---

### DNN Classification (no TensorFlow needed)

```python
import cvfiq

classifier = cvfiq.dnn('model.onnx', 'labels.txt', imgSize=(224, 224))
# classifier.useGPU()  # enable CUDA if available

def process(img):
    scores, index, confidence = classifier.getPrediction(img)
    topK = classifier.getTopK(img, k=3)  # [("cat", 0.92), ...]
    return img

cvfiq.run(process)
```

---

### Video Stabilizer

```python
import cvfiq

stab = cvfiq.stabilizer(smoothRadius=15, border='black')
# border options: 'black', 'replicate', 'reflect'

def process(img):
    return stab.stabilize(img)

cvfiq.run(process)
```

---

### Color Detection

```python
import cvfiq

cf      = cvfiq.color(trackBar=True)  # trackBar=True shows HSV sliders
hsvVals = {'hmin': 10, 'smin': 55, 'vmin': 215, 'hmax': 42, 'smax': 255, 'vmax': 255}

def process(img):
    imgColor, mask = cf.update(img, hsvVals)
    # cf.update(img, 'red')               # built-in color
    # cf.updateMulti(img, ['red','blue'])  # multiple colors
    return imgColor

cvfiq.run(process)
```

---

### FPS Counter

`cvfiq.run()` shows FPS by default (`showFPS=True`). To use manually:

```python
import cvfiq

fpsReader = cvfiq.fps(avgCount=10)

with cvfiq.Camera(0) as cam:
    for img in cam:
        f, img = fpsReader.update(img, pos=(20, 50), color=(0, 255, 0))
        cam.show(img)
```

---

### PID Controller

```python
import cvfiq

detector = cvfiq.face()
xPID = cvfiq.pid([1, 0.0001, 0.5], targetVal=320, iLimit=[-500, 500])
yPID = cvfiq.pid([1, 0.0001, 0.5], targetVal=240, axis=1, limit=[-100, 100], iLimit=[-500, 500])

def process(img):
    img, bboxs = detector.findFaces(img)
    if bboxs:
        cx, cy = bboxs[0]["center"]
        xPID.update(cx);  xPID.draw(img, [cx, cy])
        yPID.update(cy);  yPID.draw(img, [cx, cy])
    else:
        xPID.reset(); yPID.reset()
    return img

cvfiq.run(process)
```

---

### Live Plot

```python
import cvfiq, math

xPlot = cvfiq.plot(w=640, h=480, yLimit=[-100, 100])
x     = 0

while True:
    x = (x + 1) % 360
    imgPlot = xPlot.update(int(math.sin(math.radians(x)) * 100))
    cvfiq.imshow("Plot", imgPlot)
    if cvfiq.waitKey(1) & 0xFF == ord('q'):
        break
cvfiq.destroyAllWindows()
```

---

### Stack Images

```python
import cvfiq

def process(img):
    imgFlip = cvfiq.flip(img, 1)
    imgList = [img, imgFlip, img, imgFlip, img, imgFlip]
    return cvfiq.stackImages(imgList, cols=3, scale=0.4)

cvfiq.run(process)
```

---

### Arduino Serial Communication

```python
import cvfiq, time

arduino = cvfiq.serial(portNo=None, baudRate=9600, digits=3, timeout=1)

while True:
    arduino.sendData([1, 1, 0])   # sends "$001001000"
    data = arduino.getData()
    print(data)
    time.sleep(0.1)
```

---

### Classification (Teachable Machine)

```python
import cvfiq

classifier = cvfiq.classify('Model/keras_model.h5', 'Model/labels.txt', imgSize=224)

def process(img):
    scores, index, confidence = classifier.getPrediction(img)
    print(f"Predicted: index={index}  confidence={confidence:.2f}")
    return img

cvfiq.run(process)
```

---

## Utility Functions

```python
import cvfiq

img = cvfiq.imread("image.jpg")

# Stack multiple images into one window
stacked = cvfiq.stackImages([img, img, img], cols=3, scale=0.5)

# Fancy bounding box with corner lines
img = cvfiq.cornerRect(img, bbox=(100, 100, 200, 150))

# Find contours with area filter
conFound, imgCon = cvfiq.findContours(img, imgPre, minArea=1000, filter=4)

# Overlay transparent PNG
img = cvfiq.overlayPNG(imgBack=img, imgFront=imgPNG, pos=[50, 50])

# Rotate image
img = cvfiq.rotateImage(img, angle=45, scale=1)

# Text with background rectangle
img, rect = cvfiq.putTextRect(img, "Hello", pos=(50, 50), scale=2, thickness=2)

cvfiq.imwrite("output.jpg", img)
```

---

### QR Code & Barcode Detection

```python
import cvfiq

scanner = cvfiq.qr()

def process(img):
    codes, img = scanner.findCodes(img)
    for c in codes:
        print(c["data"], c["type"])
        # keys: "data", "type", "corners", "center"
    return img

cvfiq.run(process)
```

---

### Motion Detection

```python
import cvfiq

detector = cvfiq.motion(threshold=500, history=200)

def process(img):
    detected, regions, img = detector.findMotion(img)
    if detected:
        print(f"Motion in {len(regions)} region(s)")
    return img

cvfiq.run(process)
```

---

### Object Tracker

```python
import cvfiq

tracker = cvfiq.tracker(algo='CSRT')  # CSRT (default), KCF, MIL

def process(img):
    success, bbox, img = tracker.update(img)
    return img

# Initialize tracker with bounding box — select ROI with mouse
with cvfiq.Camera(0) as cam:
    for img in cam:
        bbox = tracker.select(img)   # opens ROI selector
        break
cvfiq.run(process)
```

---

### OCR (Text Detection)

Requires: `pip install pytesseract` + [tesseract binary](https://github.com/tesseract-ocr/tesseract)

```python
import cvfiq

reader = cvfiq.ocr()

def process(img):
    texts, img = reader.findText(img)
    for t in texts:
        print(t["text"], t["confidence"])
        # keys: "text", "confidence", "bbox", "center"
    return img

cvfiq.run(process)
```

---

### Depth Estimation

Model auto-downloaded on first use (MiDaS ONNX).

```python
import cvfiq

estimator = cvfiq.depth()

def process(img):
    depthMap, colorized = estimator.findDepth(img)
    return colorized   # colorized depth map

cvfiq.run(process)
```

---

### Emotion Detection

Requires: `pip install deepface`

```python
import cvfiq

detector = cvfiq.emotion()

def process(img):
    emotions, img = detector.findEmotions(img)
    for e in emotions:
        print(e["emotion"], e["scores"])
        # keys: "emotion", "scores", "bbox", "center"
    return img

cvfiq.run(process)
```

---

### Age & Gender Detection

Models auto-downloaded on first use.

```python
import cvfiq

detector = cvfiq.age_gender()

def process(img):
    results, img = detector.findAgeGender(img)
    for r in results:
        print(r["gender"], r["age"])
        # keys: "gender", "genderConf", "age", "ageConf", "bbox", "center"
    return img

cvfiq.run(process)
```

---

### Pipeline (chain functions)

```python
import cvfiq

hand = cvfiq.hand()
face = cvfiq.face()

pipe = cvfiq.pipeline(
    lambda img: hand.findHands(img)[1],
    lambda img: face.findFaces(img)[0],
)

cvfiq.run(pipe)
```

---

### Text & Snapshot helpers

```python
import cvfiq

def process(img):
    cvfiq.text(img, "Hello!", (10, 30))
    cvfiq.text(img, "Label", (10, 60), color=(0, 255, 0), bg=True)
    return img

# Press 's' to save snapshot
cvfiq.run(process, keys={'s': lambda img: cvfiq.snapshot(img)})
```

---

### Advanced `cvfiq.run()` parameters

```python
cvfiq.run(process, source=0)                      # default webcam
cvfiq.run(process, source=1)                      # second camera
cvfiq.run(process, source="video.mp4")            # video file
cvfiq.run(process, show=False, record="out.mp4")  # headless + save video
cvfiq.run(process, width=1280, height=720)        # set resolution
cvfiq.run(process, keys={'s': lambda img: cvfiq.snapshot(img)})  # key callbacks
```

| Parameter | Default | Description |
|---|---|---|
| `fn` | — | Function that receives `img` and returns processed `img` |
| `source` | `0` | Camera index or video file path |
| `title` | `"cvfiq"` | Window title |
| `showFPS` | `True` | Overlay FPS counter |
| `quitKey` | `'q'` | Key to exit |
| `width` / `height` | `None` | Set camera resolution |
| `show` | `True` | Display window (`False` = headless/server mode) |
| `record` | `None` | File path to save output video, e.g. `"output.mp4"` |
| `keys` | `None` | Dict of key callbacks, e.g. `{'s': lambda img: cvfiq.snapshot(img)}` |

---

## Module Overview

### Camera helpers

| Function / Class | Description |
|---|---|
| `cvfiq.run(fn, source=0, showFPS=True, title="cvfiq", quitKey='q')` | Open camera/video, call `fn(img)` each frame, display result. `source` = camera index or file path. |
| `cvfiq.Camera(source=0, showFPS=False, title="cvfiq", quitKey='q', width=None, height=None)` | Context manager for manual camera loop. Iterate for frames, call `cam.show(img)` to display. |

### Detectors

| Shorthand | Description |
|---|---|
| `cvfiq.face()` | Face detection + 6 keypoints |
| `cvfiq.mesh()` | 468 face landmarks, blink & mouth detection |
| `cvfiq.hand()` | 21 hand landmarks, finger detection, 2D/3D distance, angle |
| `cvfiq.pose()` | Body pose estimation, angle, smoothing |
| `cvfiq.segment()` | Real-time background removal with smooth alpha blending |
| `cvfiq.gesture()` | Gesture recognition — model auto-downloaded |
| `cvfiq.landmarker()` | 478 landmarks + 52 expression blendshapes — model auto-downloaded |
| `cvfiq.detector()` | 80-class object detection — model auto-downloaded |
| `cvfiq.aruco()` | ArUco marker detection + 3D pose estimation |
| `cvfiq.dnn(model, labels)` | ONNX/TFLite classification via OpenCV DNN |
| `cvfiq.stabilizer()` | Real-time video stabilization |
| `cvfiq.color()` | HSV color detection, save/load profiles, multi-color |
| `cvfiq.classify(model, labels)` | Teachable Machine / Keras model inference |
| `cvfiq.fps()` | Averaged FPS counter |
| `cvfiq.pid(pidVals, targetVal)` | PID controller with integral windup clamp |
| `cvfiq.plot()` | Real-time live graph in OpenCV window |
| `cvfiq.serial()` | Arduino serial communication |
| `cvfiq.qr()` | QR code & barcode detection |
| `cvfiq.motion()` | Background subtraction motion detection |
| `cvfiq.tracker(algo)` | Single-object tracker (CSRT/KCF/MIL) |
| `cvfiq.ocr()` | Text recognition via pytesseract *(optional)* |
| `cvfiq.depth()` | MiDaS monocular depth estimation — model auto-downloaded *(optional)* |
| `cvfiq.emotion()` | Emotion recognition via deepface *(optional)* |
| `cvfiq.age_gender()` | Age & gender estimation — models auto-downloaded *(optional)* |
| `cvfiq.pipeline(*fns)` | Chain multiple process functions into one |
| `cvfiq.text(img, txt, pos)` | One-line putText shorthand |
| `cvfiq.snapshot(img)` | Save numbered JPEG file |
| `cvfiq.stackImages()`, `cvfiq.cornerRect()`, … | Utility + bundled cv2 functions (`imread`, `imshow`, `circle`, …) |

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

# New module tests (webcam required)
python test_qr.py
python test_motion.py
python test_tracker.py

# Optional module tests (extra deps needed)
python test_ocr.py       # pip install pytesseract
python test_depth.py     # model auto-downloaded
python test_emotion.py   # pip install deepface
python test_age_gender.py  # models auto-downloaded
```

---

## License

MIT License — see [LICENSE](LICENSE)
