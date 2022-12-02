import cv2 as cv
from HandDetector import HandDetector
from Preprocessor import Preprocessor

class Manager:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.frameWidth = self.cap.get(cv. CAP_PROP_FRAME_WIDTH)
        self.frameHeight = self.cap.get(cv. CAP_PROP_FRAME_HEIGHT)
        self.handDetector = HandDetector(self.frameWidth, self.frameHeight)
        self.preprocessor = Preprocessor()

    def run(self):

        while True:
            ret, frame = self.cap.read()

            frame = self.preprocessor.run(frame)

            frame, skinMask = self.handDetector.run(frame)
            cv.imshow("video", frame)
            cv.imshow("mask", skinMask)

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

