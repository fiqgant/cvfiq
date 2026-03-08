import cv2
import mediapipe as mp
import numpy as np


class SelfiSegmentation():

    def __init__(self, model=1):
        """
        :param model: model type 0 or 1. 0 is general 1 is landscape(faster)
        """
        self.model = model
        self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
        self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(model_selection=self.model)

    def removeBG(self, img, imgBg=(255, 255, 255), threshold=0.1, smooth=True, kernelSize=11):
        """
        :param img: image to remove background from
        :param imgBg: BackGround Image (tuple color or image array)
        :param threshold: higher = more cut, lower = less cut
        :param smooth: blur mask edges for smoother transition
        :param kernelSize: blur kernel size for smoothing (odd number)
        :return: image with background replaced
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.selfieSegmentation.process(imgRGB)
        mask = results.segmentation_mask

        if smooth:
            mask = cv2.GaussianBlur(mask, (kernelSize, kernelSize), 0)

        # Alpha blend using soft mask instead of hard threshold
        alpha = np.clip((mask - threshold) / (1.0 - threshold), 0.0, 1.0)
        alpha3 = np.stack([alpha] * 3, axis=-1).astype(np.float32)

        if isinstance(imgBg, tuple):
            _imgBg = np.zeros(img.shape, dtype=np.uint8)
            _imgBg[:] = imgBg
        else:
            _imgBg = imgBg

        imgF = img.astype(np.float32)
        bgF  = _imgBg.astype(np.float32)
        imgOut = (imgF * alpha3 + bgF * (1.0 - alpha3)).astype(np.uint8)
        return imgOut


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    segmentor = SelfiSegmentation()
    imgBg = cv2.imread("background.jpg")
    while True:
        success, img = cap.read()
        imgOut = segmentor.removeBG(img, imgBg=imgBg, threshold=0.1)

        cv2.imshow("Image", img)
        cv2.imshow("Image Out", imgOut)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()

