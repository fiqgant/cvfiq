"""
EmotionModule — Facial emotion detection.
Requires: pip install deepface
DeepFace handles its own model downloads automatically.
"""

import cv2


class EmotionDetector:
    """
    Detect facial emotions: happy, sad, angry, fear, disgust, surprise, neutral.

    Usage:
        emotion = EmotionDetector()
        results, img = emotion.findEmotions(img)
        for r in results:
            print(r["emotion"], r["scores"])
    """

    def __init__(self, enforce_detection=False):
        """
        :param enforce_detection: If True, raises error when no face found.
                                  Default False (graceful handling).
        """
        self._enforce = enforce_detection
        self._df      = None
        try:
            from deepface import DeepFace
            self._df = DeepFace
        except ImportError:
            pass

    @property
    def available(self):
        return self._df is not None

    def findEmotions(self, img, draw=True):
        """
        Detect emotions in all faces found in img.

        Returns:
            results (list): [{
                "emotion": str,          # dominant emotion
                "scores": dict,          # {"happy": 0.95, "sad": 0.01, ...}
                "bbox": (x, y, w, h),
                "center": (cx, cy)
            }]
            img: annotated image
        """
        if self._df is None:
            if draw:
                cv2.putText(img, "Emotion: pip install deepface", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return [], img

        try:
            raw = self._df.analyze(
                img,
                actions=['emotion'],
                enforce_detection=self._enforce,
                silent=True,
            )
        except Exception:
            return [], img

        _COLORS = {
            'happy':    (0, 220, 0),
            'sad':      (255, 100, 0),
            'angry':    (0, 0, 255),
            'fear':     (128, 0, 128),
            'disgust':  (0, 128, 128),
            'surprise': (0, 200, 255),
            'neutral':  (180, 180, 180),
        }

        results = []
        for face in raw:
            emotion = face.get('dominant_emotion', 'neutral')
            scores  = face.get('emotion', {})
            region  = face.get('region', {})
            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('w', 0)
            h = region.get('h', 0)
            cx, cy = x + w // 2, y + h // 2
            color = _COLORS.get(emotion, (0, 255, 0))

            results.append({
                "emotion": emotion,
                "scores":  scores,
                "bbox":    (x, y, w, h),
                "center":  (cx, cy),
            })

            if draw:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, emotion.upper(), (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                top_score = scores.get(emotion, 0)
                cv2.putText(img, f"{top_score:.0f}%", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return results, img
