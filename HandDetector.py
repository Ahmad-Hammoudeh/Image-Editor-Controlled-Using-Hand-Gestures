import logging
import traceback

import cv2 as cv
import numpy as np
import threading
import keyboard


class HandDetector:
    def __init__(self, cap):
        self.cap = cap
        self.frameWidth = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        self.referenceFrame = np.full((self.frameHeight, self.frameWidth, 1), 0, np.uint8)
        self.backGroundThread = threading.Thread(target=self.takeBackgroundFrame1)
        self.backGroundThread.start()

        self.isCalibrated = False
        self.doCalibration = False
        self.calibrationThread = threading.Thread(target=self.calibrationTrigger)
        self.calibrationThread.start()

        self.hueLow, self.hueHigh = 0, 0
        self.satLow, self.satHigh = 0, 0
        self.valueLow, self.valueHigh = 0, 0

        self.background = np.full((self.frameHeight, self.frameWidth, 1), 0, np.uint8)
        self.accumulatedWeight = 0.5
        self.BGThreshold = 10

    def run(self, originalFrame):
        handPixels = np.copy(originalFrame)

        handPixels = self.clearBackground1(handPixels)

        '''
        if self.doCalibration:
            self.calThreshold(handPixels)
            self.doCalibration = False
            self.isCalibrated = True

        handPixels = self.generateSkinMask(handPixels)

        if not self.isCalibrated:
            self.drawSamplingRecs(originalFrame)
        '''
        handPixels = self.handArea(handPixels, originalFrame)
        DFrame = np.copy(handPixels)
        DFrame, handContour = self.drawHandContour(DFrame)

        return handPixels, DFrame, handContour

    def generateSkinMask(self, frame):
        hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsvFrame, (self.hueLow, self.satLow, self.valueLow),
                          (self.hueHigh, self.satHigh, self.valueHigh))

        # Opening using 'ellipse shape' array
        ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, ellipse)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, ellipse)
        # Dilation
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.dilate(mask, kernel, iterations=2)

        # blurred = cv.bilateralFilter(mask, 9, 75, 75)
        # edges = cv.Canny(blurred, 10, 100)
        # kernel = np.ones((3, 3), np.uint8)
        # edges = cv.dilate(edges, kernel, iterations=2)
        # cv.imshow("edges" ,edges)

        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        return cv.bitwise_and(frame, mask)

    def calibrationTrigger(self):
        while True:
            keyboard.wait("c")
            self.doCalibration = True
            keyboard.wait("c")
            self.doCalibration = False
            self.isCalibrated = False

    def calThreshold(self, frame):
        samplingRec1 = frame[int(self.frameHeight * 0.35) : int(self.frameHeight * 0.35) + 20,
                       int(self.frameWidth * 0.75):int(self.frameWidth * 0.75) + 20]

        # Convert to HSV
        samplingRec1 = cv.cvtColor(samplingRec1, cv.COLOR_BGR2HSV)

        samplingRec2 = frame[int(self.frameHeight * 0.50):int(self.frameHeight * 0.50) + 20,
                       int(self.frameWidth * 0.75):int(self.frameWidth * 0.75) + 20]

        samplingRec2 = cv.cvtColor(samplingRec2, cv.COLOR_BGR2HSV)

        offsetLowThreshold = 80
        offsetHighThreshold = 30
        # offsetLowThreshold = 80
        # offsetHighThreshold = 30

        meanRec1 = np.average(samplingRec1, axis=(0, 1))
        meanRec2 = np.average(samplingRec2, axis=(0, 1))

        self.hueLow = min(meanRec1[0], meanRec2[0]) - offsetLowThreshold
        self.hueHigh = max(meanRec1[0], meanRec2[0]) + offsetHighThreshold

        self.satLow = min(meanRec1[1], meanRec2[1]) - offsetLowThreshold
        self.satHigh = max(meanRec1[1], meanRec2[1]) + offsetHighThreshold

        self.valueLow = min(meanRec1[2], meanRec2[2]) - offsetLowThreshold
        self.valueHigh = max(meanRec1[2], meanRec2[2]) + offsetHighThreshold

        # print(self.hueLow, self.hueHigh)
        # print(self.satLow, self.satHigh)
        # print(self.valueLow, self.valueHigh)

    def clearBackground2(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_blur = cv.GaussianBlur(gray, (7, 7), 75)

        diff = cv.absdiff(self.background.astype('uint8'), gray_blur)

        ret, mask = cv.threshold(diff, self.BGThreshold, 255, cv.THRESH_BINARY)

        ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, ellipse)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv.dilate(mask, kernel, iterations=2)
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        cv.imshow("hand mask", mask)
        return cv.bitwise_and(frame, mask)

    def takeBackgroundFrame2(self):
        while True:
            keyboard.wait("b")

            self.background = np.full((self.frameHeight, self.frameWidth, 1), 0, np.uint8)

            curBackground = None
            cnt = 0
            while cnt < 60:
                ret, frame = self.cap.read()
                frame = cv.flip(frame, 1)
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                gray_blur = cv.GaussianBlur(gray, (7, 7), 75)

                if cnt == 0:
                    curBackground = gray_blur.copy().astype('float')
                else:
                    cv.accumulateWeighted(gray_blur, curBackground, self.accumulatedWeight)

                cnt += 1

            self.background = curBackground

    def clearBackground1(self, frame):
        grayFrame = None
        try:
            thresholdOffset = 15
            grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            dif = cv.absdiff(grayFrame, self.referenceFrame)
            ret, dif = cv.threshold(dif, thresholdOffset + 1, 255, cv.THRESH_BINARY)

            ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13, 13))
            dif = cv.morphologyEx(dif, cv.MORPH_OPEN, ellipse)
            dif = cv.morphologyEx(dif, cv.MORPH_CLOSE, ellipse)

            kernel = np.ones((5, 5), np.uint8)
            backgroundFrame = cv.dilate(dif, kernel, iterations=2)

            backgroundFrame = cv.cvtColor(backgroundFrame, cv.COLOR_GRAY2BGR)

            # backgroundFrame = cv.cvtColor(dif, cv.COLOR_GRAY2BGR)
            cv.imshow("test", backgroundFrame)
            return cv.bitwise_and(frame, backgroundFrame)

        except Exception as e:
            logging.error(traceback.format_exc())
            raise Exception("tt")

    def takeBackgroundFrame1(self):
        while True:
            keyboard.wait("b")
            ret, self.referenceFrame = self.cap.read()
            self.referenceFrame = cv.flip(self.referenceFrame, 1)
            # self.referenceFrame = cv.GaussianBlur(self.referenceFrame, (5, 5), 75)
            self.referenceFrame = cv.cvtColor(self.referenceFrame, cv.COLOR_BGR2GRAY)

    def drawHandContour(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        # blurred = cv.bilateralFilter(gray, 9, 75, 75)
        # edges = cv.Canny(blurred, 10, 100)
        # kernel = np.ones((3, 3), np.uint8)
        # edges = cv.dilate(edges, kernel, iterations=2)
        # cv.imshow("edges" ,edges)
        contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        maxContour = []
        if len(contours) > 0:
            maxContour = max(contours, key=cv.contourArea)
        cv.drawContours(frame, maxContour, -1, (0, 255, 0), 2)
        # cv.drawContours(frame, contours, -1, (0, 255, 0), 2)

        return cv.cvtColor(frame, cv.COLOR_RGB2BGR), maxContour

    def handArea(self, frame, originalFrame):
        # pt1 = (int(self.frameWidth * 0.55), 0)
        # pt2 = (int(self.frameWidth * 0.55), self.frameHeight)
        # pt1 = tuple([int(round(pt1[0])), int(round(pt1[1]))])
        # pt2 = tuple([int(round(pt2[0])), int(round(pt2[1]))])
        # cv.line(originalFrame, pt1, pt2, (0, 0, 255), 2)
        #
        # frame[:, 0:int(self.frameWidth * 0.54) - 1] = 0

        minH = 0
        maxH = int(self.frameHeight * 0.75)
        minW = int(self.frameWidth * 0.50)
        maxW = self.frameWidth

        pt1 = (minW, minH)
        pt2 = (maxW, maxH)
        # print("X:", (maxW + minW) / 2)
        # print("Y:", (maxH+minH)/2)

        cv.rectangle(originalFrame, pt1, pt2, (0, 0, 255), 2)

        zeros = np.full((self.frameHeight, self.frameWidth, 3), 0, np.uint8)
        zeros[minH:maxH, minW:maxW] = frame[minH:maxH, minW:maxW]

        return zeros



    def drawSamplingRecs(self, frame):
        start_point1, end_point1, start_point2, end_point2 = self.samplingRecPoints()

        frame = cv.rectangle(frame, start_point1, end_point1, (255, 0, 0), 2)
        frame = cv.rectangle(frame, start_point2, end_point2, (255, 0, 0), 2)

    def samplingRecPoints(self):
        start_point1 = (int(self.frameWidth * 0.75), int(self.frameHeight * 0.35))
        end_point1 = (int(self.frameWidth * 0.75) + 20, int(self.frameHeight * 0.35) + 20)

        start_point2 = (int(self.frameWidth * 0.75), int(self.frameHeight * 0.50))
        end_point2 = (int(self.frameWidth * 0.75) + 20, int(self.frameHeight * 0.50) + 20)

        return start_point1, end_point1, start_point2, end_point2
