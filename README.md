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

### Hand Tracking

**`cvfiq.hand(maxHands=2, detectionCon=0.5, minTrackCon=0.5, modelComplexity=1)`**

Detects and tracks up to N hands with 21 landmarks each.

| Method | Returns | Description |
|---|---|---|
| `findHands(img, draw=True, flipType=True)` | `(hands, img)` | Detect hands; each hand dict has `lmList`, `bbox`, `center`, `type` |
| `fingersUp(hand)` | `[0/1 × 5]` | Which fingers are extended: `[thumb, index, middle, ring, pinky]` |
| `findDistance(p1, p2, img=None)` | `(dist, info, img)` | 2D pixel distance between two landmarks |
| `findDistance3D(p1, p2)` | `float` | 3D distance using x, y, z coordinates |
| `findAngle(p1, p2, p3, img=None)` | `float` | Angle in degrees at p2 formed by p1–p2–p3 |

```python
import cvfiq

detector = cvfiq.hand(detectionCon=0.8, maxHands=2)

def process(img):
    hands, img = detector.findHands(img)
    if hands:
        hand1   = hands[0]
        lmList  = hand1["lmList"]   # 21 landmarks [x, y, z]
        bbox    = hand1["bbox"]     # x, y, w, h
        type1   = hand1["type"]     # "Left" or "Right"

        fingers = detector.fingersUp(hand1)             # [0,1,1,0,0]
        dist, info, img = detector.findDistance(lmList[8][:2], lmList[4][:2], img)
        dist3d = detector.findDistance3D(lmList[8], lmList[4])
        angle  = detector.findAngle(lmList[5], lmList[9], lmList[13])
    return img

cvfiq.run(process)
```

---

### Face Detection

**`cvfiq.face(minDetectionCon=0.75, modelSelection=0)`**

Detects faces with bounding boxes and 6 facial keypoints.

| Method | Returns | Description |
|---|---|---|
| `findFaces(img, draw=True)` | `(img, bboxs)` | Detect faces; each bbox dict has `id`, `bbox`, `score`, `center`, `keypoints` |

`keypoints` keys: `rightEye`, `leftEye`, `nose`, `mouth`, `rightEar`, `leftEar`

```python
import cvfiq

detector = cvfiq.face()

def process(img):
    img, bboxs = detector.findFaces(img)
    if bboxs:
        cx, cy = bboxs[0]["center"]
        cvfiq.circle(img, (cx, cy), 5, (255, 0, 255), cvfiq.FILLED)
    return img

cvfiq.run(process)
```

---

### Face Mesh

**`cvfiq.mesh(maxFaces=2, refineLandmarks=True)`**

Detects 468 face landmarks with blink and mouth state detection.

| Method | Returns | Description |
|---|---|---|
| `findFaceMesh(img, draw=True)` | `(img, faces)` | Detect face landmarks; each face is a list of 468 `[x, y, z]` points |
| `getRegion(face, region)` | `list` | Landmark points for a named facial region (use `FaceMeshDetector` constants) |
| `blinkDetector(face)` | `dict` | `{"left": bool, "right": bool, "leftEAR": float, "rightEAR": float}` |
| `mouthOpen(face)` | `dict` | `{"open": bool, "ratio": float}` |
| `findDistance(p1, p2, img=None)` | `(dist, info, img)` | Distance between two landmark points |

```python
import cvfiq
from cvfiq.FaceMeshModule import FaceMeshDetector

detector = cvfiq.mesh(maxFaces=2)

def process(img):
    img, faces = detector.findFaceMesh(img)
    if faces:
        blink = detector.blinkDetector(faces[0])
        mouth = detector.mouthOpen(faces[0])
        region = detector.getRegion(faces[0], FaceMeshDetector.LEFT_EYE)
    return img

cvfiq.run(process)
```

---

### Pose Estimation

**`cvfiq.pose(smooth=True, smoothAlpha=0.5, detectionCon=0.5, trackCon=0.5)`**

Detects 33 full-body pose landmarks.

| Method | Returns | Description |
|---|---|---|
| `findPose(img, draw=True)` | `img` | Detect and draw pose skeleton |
| `findPosition(img, draw=True, bboxWithHands=False)` | `(lmList, bboxInfo)` | Get landmark coordinates and bounding box |
| `findAngle(img, p1, p2, p3, draw=True)` | `float` | Angle in degrees at p2 between three landmark indices |
| `findDistance(p1, p2, img, draw=True)` | `(dist, info, img)` | Distance between two landmark indices |
| `angleCheck(myAngle, targetAngle, addOn=20)` | `bool` | True if angle is within `targetAngle ± addOn` |

```python
import cvfiq

detector = cvfiq.pose(smoothAlpha=0.5)

def process(img):
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)
    if lmList:
        angle = detector.findAngle(img, 11, 13, 15)   # shoulder-elbow-wrist
        ok    = detector.angleCheck(angle, targetAngle=90, addOn=20)
    return img

cvfiq.run(process)
```

---

### Selfie Segmentation (Background Removal)

**`cvfiq.segment(model=1)`** — `model=0` general, `model=1` landscape (faster)

| Method | Returns | Description |
|---|---|---|
| `removeBG(img, imgBg, threshold=0.1, smooth=True, kernelSize=11)` | `img` | Remove background; `imgBg` can be a color tuple `(R,G,B)` or an image array |

```python
import cvfiq

seg   = cvfiq.segment(model=1)
imgBg = cvfiq.imread("background.jpg")   # or use a color: (0, 255, 0)

def process(img):
    return seg.removeBG(img, imgBg=imgBg, smooth=True)

cvfiq.run(process)
```

---

### Gesture Recognition *(model auto-downloaded)*

**`cvfiq.gesture()`**

Recognizes 8 hand gestures using MediaPipe Tasks API.

| Method | Returns | Description |
|---|---|---|
| `findGestures(img, draw=True)` | `(gestures, img)` | Detect gestures; each dict has `gesture`, `score`, `hand`, `lmList` |

Supported gestures: `Thumb_Up`, `Thumb_Down`, `Victory`, `Open_Palm`, `Closed_Fist`, `Pointing_Up`, `ILoveYou`, `None`

```python
import cvfiq

detector = cvfiq.gesture()   # model auto-downloaded on first run

def process(img):
    gestures, img = detector.findGestures(img)
    for g in gestures:
        print(g["hand"], g["gesture"], g["score"])
    return img

cvfiq.run(process)
```

---

### Face Landmarker + Expressions *(model auto-downloaded)*

**`cvfiq.landmarker(maxFaces=1, outputBlendshapes=True)`**

Detects 478 face landmarks and 52 expression blendshapes.

| Method | Returns | Description |
|---|---|---|
| `findFaces(img, draw=True)` | `(faces, img)` | Detect face landmarks and blendshapes |
| `getBlendshape(faceInfo, name)` | `float` | Score 0–1 for a named blendshape (use `FaceLandmarker` constants) |
| `isSmiling(faceInfo, threshold=0.5)` | `bool` | Smile detection |
| `isBlinking(faceInfo, eye='both', threshold=0.4)` | `bool` | Blink detection; `eye='left'/'right'/'both'` |
| `isMouthOpen(faceInfo, threshold=0.3)` | `bool` | Mouth open detection |
| `getExpression(faceInfo, threshold=0.5)` | `dict` | Active expressions above threshold: `smiling`, `mouth_open`, `blink_left`, `blink_right`, `brow_down`, `neutral` |

```python
import cvfiq
from cvfiq.FaceLandmarkerModule import FaceLandmarker

detector = cvfiq.landmarker()   # model auto-downloaded on first run

def process(img):
    faces, img = detector.findFaces(img)
    if faces:
        print(detector.getExpression(faces[0]))
        jaw = detector.getBlendshape(faces[0], FaceLandmarker.JAW_OPEN)
    return img

cvfiq.run(process)
```

---

### Object Detection *(model auto-downloaded)*

**`cvfiq.detector(maxResults=5, scoreThreshold=0.4)`**

Detects 80 COCO object classes.

| Method | Returns | Description |
|---|---|---|
| `findObjects(img, draw=True)` | `(objects, img)` | Detect objects; each dict has `label`, `score`, `bbox` (x,y,w,h), `center` |
| `getObjectsByLabel(objects, label)` | `list` | Filter objects by label name |
| `countObjects(objects, label=None)` | `int` | Count objects, optionally filtered by label |

```python
import cvfiq

detector = cvfiq.detector(scoreThreshold=0.5)   # model auto-downloaded on first run

def process(img):
    objects, img = detector.findObjects(img)
    count = detector.countObjects(objects, 'person')
    print(f"People: {count}")
    return img

cvfiq.run(process)
```

---

### ArUco Marker Detection

**`cvfiq.aruco(dictType='DICT_4X4_50')`**

| Method | Returns | Description |
|---|---|---|
| `findMarkers(img, draw=True)` | `(markers, img)` | Detect ArUco markers; each dict has `id`, `corners`, `center` |
| `estimatePose(img, markers, markerSize=0.05, cameraMatrix=None, distCoeffs=None)` | `(poses, img)` | Estimate 3D pose; each dict has `id`, `distance` (meters), `rvec`, `tvec` |
| `generateMarker(markerId, size=200, dictType='4x4_50')` | `img` | Generate and return a marker image |

```python
import cvfiq

detector = cvfiq.aruco(dictType='DICT_4X4_50')

def process(img):
    markers, img = detector.findMarkers(img)
    if markers:
        poses, img = detector.estimatePose(img, markers, markerSize=0.05)
        for pose in poses:
            print(f"ID:{pose['id']}  dist:{pose['distance']:.3f}m")
    return img

cvfiq.run(process)

# Generate and save a marker (no camera needed)
markerImg = cvfiq.aruco().generateMarker(markerId=0, size=200)
cvfiq.imwrite("marker_0.png", markerImg)
```

---

### DNN Classification (no TensorFlow needed)

**`cvfiq.dnn(modelPath, labelsPath, imgSize=(224, 224))`**

Runs ONNX or TFLite models via OpenCV DNN backend.

| Method | Returns | Description |
|---|---|---|
| `useGPU()` | — | Enable CUDA GPU acceleration |
| `getPrediction(img, draw=True)` | `(scores, index, confidence)` | Run inference; returns all class scores, top index, and confidence |
| `getTopK(img, k=3)` | `[(label, score), …]` | Top-K predictions with labels and scores |

```python
import cvfiq

classifier = cvfiq.dnn('model.onnx', 'labels.txt', imgSize=(224, 224))
# classifier.useGPU()   # enable CUDA if available

def process(img):
    scores, index, confidence = classifier.getPrediction(img)
    topK = classifier.getTopK(img, k=3)
    return img

cvfiq.run(process)
```

---

### Video Stabilizer

**`cvfiq.stabilizer(smoothRadius=15, maxCorners=200, border='black')`**

Real-time video stabilization using optical flow. `border`: `'black'`, `'replicate'`, `'reflect'`.

| Method | Returns | Description |
|---|---|---|
| `stabilize(img)` | `img` | Stabilize a single frame |
| `getSmoothness()` | `float` | Smoothness score: `0.0` (shaky) → `1.0` (smooth) |
| `reset()` | — | Reset state when switching video source |

```python
import cvfiq

stab = cvfiq.stabilizer(smoothRadius=15, border='black')

def process(img):
    return stab.stabilize(img)

cvfiq.run(process)
```

---

### Color Detection

**`cvfiq.color(trackBar=False)`**

Detects colors by HSV range with optional live trackbar tuning.

| Method | Returns | Description |
|---|---|---|
| `update(img, myColor)` | `(imgColor, mask)` | Detect a single color; `myColor` is an HSV dict or built-in name |
| `updateMulti(img, colorList)` | `list` | Detect multiple colors in one call |
| `getColorHSV(myColor)` | `dict` | Get HSV range for a built-in or custom color |
| `saveColor(name, hsvVals, filepath)` | — | Save custom color profile to JSON |
| `loadColor(name, filepath)` | `dict` | Load custom color profile from JSON |

Built-in colors: `'red'`, `'orange'`, `'yellow'`, `'green'`, `'blue'`, `'purple'`, `'white'`

```python
import cvfiq

cf      = cvfiq.color(trackBar=True)   # trackBar=True shows HSV sliders
hsvVals = {'hmin': 10, 'smin': 55, 'vmin': 215, 'hmax': 42, 'smax': 255, 'vmax': 255}

def process(img):
    imgColor, mask = cf.update(img, hsvVals)
    # cf.update(img, 'red')                   # built-in color
    # cf.updateMulti(img, ['red', 'blue'])    # multiple colors
    return imgColor

cvfiq.run(process)
```

---

### QR Code & Barcode Detection

**`cvfiq.qr(detectBarcode=True)`**

| Method | Returns | Description |
|---|---|---|
| `find(img, draw=True)` | `(codes, img)` | Detect QR codes and barcodes; each dict has `data`, `type`, `corners`, `center` |

```python
import cvfiq

scanner = cvfiq.qr()

def process(img):
    codes, img = scanner.find(img)
    for c in codes:
        print(c["type"], c["data"])
    return img

cvfiq.run(process)
```

---

### Motion Detection

**`cvfiq.motion(minArea=500, history=500, varThreshold=16, blurSize=21)`**

Detects motion regions using MOG2 background subtraction.

| Method | Returns | Description |
|---|---|---|
| `find(img, draw=True)` | `(detected, regions, img)` | Detect motion; each region dict has `bbox` (x,y,w,h), `center`, `area` |
| `getMask(img)` | `mask` | Return raw foreground mask (grayscale) |
| `reset()` | — | Reset background model |

```python
import cvfiq

detector = cvfiq.motion(minArea=500)

def process(img):
    detected, regions, img = detector.find(img)
    if detected:
        print(f"Motion in {len(regions)} region(s)")
    return img

cvfiq.run(process)
```

---

### Object Tracker

**`cvfiq.tracker(algo='CSRT')`** — `algo`: `'CSRT'` (default, accurate), `'KCF'` (fast), `'MIL'`

| Method | Returns | Description |
|---|---|---|
| `select(img, winName=…)` | `(x, y, w, h)` | Open ROI selector window; returns chosen bounding box |
| `init(img, bbox)` | — | Initialize tracker with bounding box `(x, y, w, h)` |
| `update(img, draw=True)` | `(success, bbox, img)` | Update tracker for current frame |
| `reset()` | — | Reset tracker state |

```python
import cvfiq

tracker = cvfiq.tracker(algo='CSRT')

# Select ROI interactively then track
with cvfiq.Camera(0) as cam:
    for img in cam:
        bbox = tracker.select(img)
        if bbox:
            break

def process(img):
    success, bbox, img = tracker.update(img)
    return img

cvfiq.run(process)
```

---

### OCR (Text Recognition)

Requires: `pip install pytesseract` + [tesseract binary](https://github.com/tesseract-ocr/tesseract)

**`cvfiq.ocr(lang='eng', minConf=60, psm=11)`**

| Method | Returns | Description |
|---|---|---|
| `find(img, draw=True)` | `(texts, img)` | Detect and read text; each dict has `text`, `confidence`, `bbox`, `center` |
| `readText(img)` | `str` | Read all text from image as a single string |

```python
import cvfiq

reader = cvfiq.ocr()

def process(img):
    texts, img = reader.find(img)
    for t in texts:
        print(f"[{t['confidence']:.0f}%] {t['text']}")
    return img

cvfiq.run(process)
```

---

### Depth Estimation

Model auto-downloaded on first use (MiDaS ONNX).

**`cvfiq.depth()`**

| Method | Returns | Description |
|---|---|---|
| `find(img)` | `(depthMap, colorized)` | Estimate depth; `depthMap` is normalized float32, `colorized` is COLORMAP_MAGMA visualization |
| `getDistance(depthMap, point)` | `float` | Relative depth value at pixel `(x, y)` |
| `overlay(img, alpha=0.5)` | `img` | Blend depth heatmap onto original image |

```python
import cvfiq

estimator = cvfiq.depth()   # model auto-downloaded on first run

def process(img):
    depthMap, colorized = estimator.find(img)
    return colorized   # show depth heatmap

cvfiq.run(process)
```

---

### Emotion Detection

Requires: `pip install deepface`

**`cvfiq.emotion(enforce_detection=False)`**

| Method | Returns | Description |
|---|---|---|
| `find(img, draw=True)` | `(emotions, img)` | Detect emotions for all faces; each dict has `emotion`, `scores`, `bbox`, `center` |

Emotions: `happy`, `sad`, `angry`, `fear`, `surprise`, `disgust`, `neutral`

```python
import cvfiq

detector = cvfiq.emotion()

def process(img):
    emotions, img = detector.find(img)
    for e in emotions:
        print(e["emotion"])
    return img

cvfiq.run(process)
```

---

### Age & Gender Detection

Models auto-downloaded on first use (OpenCV DNN Caffe).

**`cvfiq.age_gender()`**

| Method | Returns | Description |
|---|---|---|
| `find(img, draw=True)` | `(results, img)` | Estimate age and gender; each dict has `gender`, `genderConf`, `age`, `ageConf`, `bbox`, `center` |

```python
import cvfiq

detector = cvfiq.age_gender()   # models auto-downloaded on first run

def process(img):
    results, img = detector.find(img)
    for r in results:
        print(f"{r['gender']} ({r['genderConf']:.0%}), age {r['age']}")
    return img

cvfiq.run(process)
```

---

### Classification (Teachable Machine / Keras)

Requires: `pip install cvfiq[keras]` or `pip install cvfiq[tensorflow]`

**`cvfiq.classify(modelPath, labelsPath, imgSize=224)`**

| Method | Returns | Description |
|---|---|---|
| `getPrediction(img, draw=True)` | `(scores, index, confidence)` | Run inference; returns all class scores, top index, and confidence |

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

### FPS Counter

**`cvfiq.fps(avgCount=10)`**

`cvfiq.run()` shows FPS by default (`showFPS=True`). To use manually:

| Method | Returns | Description |
|---|---|---|
| `update(img=None, pos=(20,50), color=(255,0,0), scale=3, thickness=3)` | `(fps, img)` or `float` | Update and return FPS; draws on image if `img` is provided |

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

**`cvfiq.pid(pidVals, targetVal, axis=0, limit=None, iLimit=None)`**

| Method | Returns | Description |
|---|---|---|
| `update(cVal)` | `float` | Compute PID output for current value |
| `reset()` | — | Reset integral and error |
| `draw(img, cVal)` | — | Draw target lines and current value on image |

```python
import cvfiq

xPID = cvfiq.pid([1, 0.0001, 0.5], targetVal=320, iLimit=[-500, 500])
yPID = cvfiq.pid([1, 0.0001, 0.5], targetVal=240, axis=1, limit=[-100, 100])

detector = cvfiq.face()

def process(img):
    img, bboxs = detector.findFaces(img)
    if bboxs:
        cx, cy = bboxs[0]["center"]
        xPID.update(cx); xPID.draw(img, [cx, cy])
        yPID.update(cy); yPID.draw(img, [cx, cy])
    else:
        xPID.reset(); yPID.reset()
    return img

cvfiq.run(process)
```

---

### Live Plot

**`cvfiq.plot(w=640, h=480, yLimit=[-100, 100])`**

| Method | Returns | Description |
|---|---|---|
| `update(y, color=(255, 0, 255))` | `img` | Add new value and return updated plot image |

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

### Arduino Serial Communication

**`cvfiq.serial(portNo=None, baudRate=9600, digits=3, timeout=1)`**

| Method | Returns | Description |
|---|---|---|
| `sendData(data)` | — | Send list of values as formatted string (e.g. `[1,1,0]` → `"$001001000"`) |
| `getData(timeout=None)` | `str` | Receive a line from serial device |

```python
import cvfiq, time

arduino = cvfiq.serial(portNo=None, baudRate=9600, digits=3, timeout=1)

while True:
    arduino.sendData([1, 1, 0])
    data = arduino.getData()
    print(data)
    time.sleep(0.1)
```

---

## Utility Functions

### Stack Images

**`cvfiq.stackImages(imgList, cols, scale)`**

Arrange multiple images into a grid. Images can be different sizes.

```python
import cvfiq

def process(img):
    imgFlip = cvfiq.flip(img, 1)
    return cvfiq.stackImages([img, imgFlip, img, imgFlip], cols=2, scale=0.5)

cvfiq.run(process)
```

---

### Pipeline

**`cvfiq.pipeline(*fns)`** — Chain multiple process functions into one.

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

### Text & Snapshot

**`cvfiq.text(img, txt, pos, color=(255,255,255), scale=1, thickness=2, bg=False)`**

**`cvfiq.snapshot(img, prefix="snap", folder=".")`** — Saves `snap_001.jpg`, `snap_002.jpg`, …

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

### Other Utilities

```python
import cvfiq

img = cvfiq.imread("image.jpg")

# Fancy bounding box with corner lines
img = cvfiq.cornerRect(img, bbox=(100, 100, 200, 150), l=30, t=5)

# Find contours with area filter
conFound, imgCon = cvfiq.findContours(img, mask, minArea=1000, filter=4)

# Overlay transparent PNG
img = cvfiq.overlayPNG(imgBack=img, imgFront=imgPNG, pos=[50, 50])

# Rotate image
img = cvfiq.rotateImage(img, angle=45, scale=1)

# Text with background rectangle
img, rect = cvfiq.putTextRect(img, "Hello", pos=(50, 50), scale=2, thickness=2)

cvfiq.imwrite("output.jpg", img)
```

---

## Advanced `cvfiq.run()` Parameters

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

```python
cvfiq.run(process, source=0)                       # default webcam
cvfiq.run(process, source=1)                       # second camera
cvfiq.run(process, source="video.mp4")             # video file
cvfiq.run(process, show=False, record="out.mp4")   # headless + save video
cvfiq.run(process, width=1280, height=720)         # set resolution
cvfiq.run(process, keys={'s': lambda img: cvfiq.snapshot(img)})
```

---

## Module Overview

### Camera helpers

| Function / Class | Description |
|---|---|
| `cvfiq.run(fn, source=0, showFPS=True, …)` | Open camera/video, call `fn(img)` each frame, display result |
| `cvfiq.Camera(source=0, showFPS=False, …)` | Context manager for manual camera loop |

### Detectors

| Shorthand | Description |
|---|---|
| `cvfiq.face()` | Face detection + 6 keypoints |
| `cvfiq.mesh()` | 468 face landmarks, blink & mouth detection |
| `cvfiq.hand()` | 21 hand landmarks, finger detection, 2D/3D distance, angle |
| `cvfiq.pose()` | 33 body landmarks, angle, distance, smoothing |
| `cvfiq.segment()` | Real-time background removal with smooth alpha blending |
| `cvfiq.gesture()` | Gesture recognition — model auto-downloaded |
| `cvfiq.landmarker()` | 478 landmarks + 52 expression blendshapes — model auto-downloaded |
| `cvfiq.detector()` | 80-class object detection — model auto-downloaded |
| `cvfiq.aruco()` | ArUco marker detection + 3D pose estimation |
| `cvfiq.dnn(model, labels)` | ONNX/TFLite classification via OpenCV DNN |
| `cvfiq.stabilizer()` | Real-time video stabilization |
| `cvfiq.color()` | HSV color detection, save/load profiles, multi-color |
| `cvfiq.classify(model, labels)` | Teachable Machine / Keras model inference |
| `cvfiq.qr()` | QR code & barcode detection |
| `cvfiq.motion()` | Background subtraction motion detection |
| `cvfiq.tracker(algo)` | Single-object tracker: CSRT / KCF / MIL |
| `cvfiq.ocr()` | Text recognition via pytesseract *(optional)* |
| `cvfiq.depth()` | MiDaS monocular depth estimation — model auto-downloaded *(optional)* |
| `cvfiq.emotion()` | Emotion recognition via deepface *(optional)* |
| `cvfiq.age_gender()` | Age & gender estimation — models auto-downloaded *(optional)* |
| `cvfiq.fps()` | Averaged FPS counter |
| `cvfiq.pid(pidVals, targetVal)` | PID controller with integral windup clamp |
| `cvfiq.plot()` | Real-time live graph in OpenCV window |
| `cvfiq.serial()` | Arduino serial communication |

### Helpers & Utilities

| Function | Description |
|---|---|
| `cvfiq.pipeline(*fns)` | Chain multiple process functions into one |
| `cvfiq.text(img, txt, pos)` | One-line putText shorthand with optional background |
| `cvfiq.snapshot(img)` | Save numbered JPEG snapshot |
| `cvfiq.stackImages(imgs, cols, scale)` | Stack images into a grid |
| `cvfiq.cornerRect(img, bbox)` | Bounding box with corner markers |
| `cvfiq.findContours(img, mask)` | Find and filter contours by area |
| `cvfiq.overlayPNG(back, front, pos)` | Overlay transparent PNG |
| `cvfiq.rotateImage(img, angle)` | Rotate image by angle |
| `cvfiq.putTextRect(img, text, pos)` | Text with background rectangle |
| `cvfiq.imread`, `cvfiq.imshow`, `cvfiq.circle`, … | Bundled cv2 functions — no `import cv2` needed |

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
python test_qr.py
python test_motion.py
python test_tracker.py

# Tasks API tests — model auto-downloaded if not present
python test_gesture.py
python test_face_landmarker.py
python test_object_detector.py

# Optional module tests (extra deps needed)
python test_ocr.py         # pip install pytesseract
python test_depth.py       # model auto-downloaded
python test_emotion.py     # pip install deepface
python test_age_gender.py  # models auto-downloaded

# GUI tests (no camera)
python test_plot.py
python test_utils.py
```

---

## License

MIT License — see [LICENSE](LICENSE)
