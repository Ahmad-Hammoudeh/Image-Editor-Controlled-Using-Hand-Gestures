import threading

import cv2 as cv
import keyboard
import numpy as np

from ImageEditor import ImageEditor
from States import States
from StateProcessor import StateProcessor

from HandDetector import HandDetector
from HandRecognition import HandRecognition
from Preprocessor import Preprocessor


class Manager:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        # ip = "http://192.168.42.129:8080/video"
        # self.cap = cv.VideoCapture(ip)
        self.frameWidth = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.frameHeight = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.handDetector = HandDetector(self.cap)
        self.preprocessor = Preprocessor()
        self.handReco = HandRecognition()
        self.currentState = States.Choose
        self.stateProcessor = StateProcessor()

        self.editor = ImageEditor()
        self.findBigImage()



        self.reloadThread = threading.Thread(target=self.reloadImage)
        self.reloadThread.start()
        # self.imageThread = threading.Thread(target=self.showImage)
        # self.imageThread.start()

    def run(self):
        while True:
            ret, frame = self.cap.read()

            frame = self.preprocessor.run(frame)
            handPixels, DFrame, handContour = self.handDetector.run(frame)

            fingersFrame, handCenterX, handCenterY, fingerCount, fingersLandMarks = \
                self.handReco.run(handPixels, frame, handContour)

            # cv.imshow("video", frame)
            # cv.imshow("hand contour", DFrame)
            # cv.imshow("hand", handPixels)

            resultFrame = self.stateProcessor.run(fingerCount, fingersFrame)

            self.bigImage = self.editor.run(handCenterX, handCenterY,
                                            fingerCount, fingersLandMarks,
                                            self.stateProcessor.state)

            cv.imshow("result", resultFrame)
            # cv.imshow("image", self.image)
            cv.imshow("image2", self.bigImage[self.Htop: self.Htop + self.h, self.Wtop: self.Wtop + self.w])

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

        cv.destroyAllWindows()
        self.cap.release()

    def findBigImage(self):
        self.image = cv.imread("home.jpg")

        self.h = self.image.shape[0]
        self.w = self.image.shape[1]

        self.H = self.h * 3
        self.W = self.w * 3
        self.bigImage = np.full((self.H, self.W, 3), 0, np.uint8)

        self.Htop = int(self.H / 2 - self.h / 2)
        self.Wtop = int(self.W / 2 - self.w / 2)

        self.bigImage[self.Htop: self.Htop + self.h, self.Wtop: self.Wtop + self.w] = self.image

        self.editor.image = self.bigImage

    def reloadImage(self):
        while True:
            keyboard.wait("r")
            # self.image = cv.imread("home.jpg")
            # self.image = np.asarray(self.image)
            # self.editor.image = self.image
            self.findBigImage()

    # def showImage(self):
    #     while True:
    #         cv.imshow('Edited Image', self.img)
    #         k = cv.waitKey(30) & 0xff
    #         if k == 27:
    #             break
