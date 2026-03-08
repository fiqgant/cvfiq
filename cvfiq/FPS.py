import time
import cv2


class FPS:
    """
    Helps in finding Frames Per Second and display on an OpenCV Image.
    Supports averaging over N frames for a stable reading.
    """

    def __init__(self, avgCount=10):
        """
        :param avgCount: Number of frames to average FPS over
        """
        self.pTime = time.time()
        self.avgCount = avgCount
        self._fpsBuffer = []

    def update(self, img=None, pos=(20, 50), color=(255, 0, 0), scale=3, thickness=3):
        """
        Update the frame rate
        :param img: Image to display on, can be left blank if only fps value required
        :param pos: Position on the FPS on the image
        :param color: Color of the FPS Value displayed
        :param scale: Scale of the FPS Value displayed
        :param thickness: Thickness of the FPS Value displayed
        :return: averaged fps, and optionally image
        """
        cTime = time.time()
        try:
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            self._fpsBuffer.append(fps)
            if len(self._fpsBuffer) > self.avgCount:
                self._fpsBuffer.pop(0)
            avgFps = sum(self._fpsBuffer) / len(self._fpsBuffer)
            if img is None:
                return avgFps
            else:
                cv2.putText(img, f'FPS: {int(avgFps)}', pos, cv2.FONT_HERSHEY_PLAIN,
                            scale, color, thickness)
                return avgFps, img
        except ZeroDivisionError:
            return 0


def main():
    """
    Without Webcam
    """
    fpsReader = FPS()
    while True:
        time.sleep(0.025)  # add delay to get 40 Frames per second
        fps = fpsReader.update()
        print(fps)


def mainWebcam():
    """
    With Webcam
    """
    fpsReader = FPS()
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        fps, img = fpsReader.update(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    # main()
    mainWebcam()
