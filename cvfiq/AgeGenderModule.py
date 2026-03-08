"""
AgeGenderModule — Age and gender estimation using OpenCV DNN (Caffe models).
Models are auto-downloaded on first use.
No extra dependencies required beyond opencv-python.
"""

import cv2
import numpy as np
import os
import urllib.request


_AGE_PROTO_URL   = ("https://raw.githubusercontent.com/smahesh29/"
                    "Gender-and-Age-Detection/master/deploy_age.prototxt")
_AGE_MODEL_URL   = ("https://github.com/smahesh29/Gender-and-Age-Detection/"
                    "raw/master/age_net.caffemodel")
_GEN_PROTO_URL   = ("https://raw.githubusercontent.com/smahesh29/"
                    "Gender-and-Age-Detection/master/deploy_gender.prototxt")
_GEN_MODEL_URL   = ("https://github.com/smahesh29/Gender-and-Age-Detection/"
                    "raw/master/gender_net.caffemodel")
_FACE_PROTO_URL  = ("https://raw.githubusercontent.com/opencv/opencv/master/"
                    "samples/dnn/face_detector/deploy.prototxt")
_FACE_MODEL_URL  = ("https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
                    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")

_AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
_GENDERS     = ['Male', 'Female']
_MEAN        = (78.4263377603, 87.7689143744, 114.895847746)


def _download(url, path):
    if os.path.exists(path):
        return
    print(f"[cvfiq] Downloading {os.path.basename(path)} ...")
    def _prog(c, b, t):
        pct = min(c * b * 100 // t, 100)
        print(f"\r[cvfiq] {os.path.basename(path)}: {pct}%", end='', flush=True)
    urllib.request.urlretrieve(url, path, reporthook=_prog)
    print()


class AgeGenderDetector:
    """
    Estimate age group and gender for faces in an image.

    Usage:
        ag = AgeGenderDetector()
        results, img = ag.find(img)
        for r in results:
            print(r["gender"], r["age"], r["center"])
    """

    def __init__(self,
                 ageProto='deploy_age.prototxt',
                 ageModel='age_net.caffemodel',
                 genderProto='deploy_gender.prototxt',
                 genderModel='gender_net.caffemodel',
                 faceProto='face_deploy.prototxt',
                 faceModel='face_res10.caffemodel',
                 faceConf=0.7):
        # Download all models if needed
        _download(_AGE_PROTO_URL,  ageProto)
        _download(_AGE_MODEL_URL,  ageModel)
        _download(_GEN_PROTO_URL,  genderProto)
        _download(_GEN_MODEL_URL,  genderModel)
        _download(_FACE_PROTO_URL, faceProto)
        _download(_FACE_MODEL_URL, faceModel)

        self._faceNet   = cv2.dnn.readNet(faceModel, faceProto)
        self._ageNet    = cv2.dnn.readNet(ageModel,  ageProto)
        self._genderNet = cv2.dnn.readNet(genderModel, genderProto)
        self._faceConf  = faceConf

    def _detectFaces(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), _MEAN,
                                     swapRB=False, crop=False)
        self._faceNet.setInput(blob)
        detections = self._faceNet.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self._faceConf:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1, conf))
        return faces

    def find(self, img, draw=True):
        """
        Estimate age and gender for all detected faces.

        Returns:
            results (list): [{
                "gender": str,        # "Male" or "Female"
                "genderConf": float,
                "age": str,           # e.g. "(25-32)"
                "ageConf": float,
                "bbox": (x,y,w,h),
                "center": (cx,cy),
                "faceConf": float,
            }]
            img: annotated image
        """
        faces = self._detectFaces(img)
        results = []

        for (x, y, w, h, fconf) in faces:
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)
            face_img = img[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), _MEAN,
                                          swapRB=False, crop=False)

            # Gender
            self._genderNet.setInput(blob)
            gPreds      = self._genderNet.forward()[0]
            genderIdx   = int(np.argmax(gPreds))
            gender      = _GENDERS[genderIdx]
            genderConf  = float(gPreds[genderIdx])

            # Age
            self._ageNet.setInput(blob)
            aPreds   = self._ageNet.forward()[0]
            ageIdx   = int(np.argmax(aPreds))
            age      = _AGE_BUCKETS[ageIdx]
            ageConf  = float(aPreds[ageIdx])

            cx, cy = x + w // 2, y + h // 2
            color  = (255, 100, 0) if gender == 'Male' else (255, 20, 147)

            results.append({
                "gender":     gender,
                "genderConf": genderConf,
                "age":        age,
                "ageConf":    ageConf,
                "bbox":       (x, y, w, h),
                "center":     (cx, cy),
                "faceConf":   fconf,
            })

            if draw:
                label = f"{gender}, {age}"
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        return results, img
