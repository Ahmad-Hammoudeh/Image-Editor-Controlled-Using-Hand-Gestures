import logging
import math
import traceback

import time

import numpy as np
import cv2 as cv
from math import *
from States import States

RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]


class ImageEditor:
    def __init__(self, image):
        self.Wtop = None
        self.Htop = None
        self.h = None
        self.w = None
        self.H = None
        self.W = None
        self.bigImage = None
        self.vector = None
        self.image = image

        self.brushMask = None
        self.drawCircleX = 0
        self.drawCircleY = 0

        self.translateCircleX = 0
        self.translateCircleY = 0

        self.brushColor = RED

        self.isEraser = False
        self.isInversedErase = True

        self.drawingList = []
        self.brushRadius = 5
        self.eraserRadius = 5

        self.findBigImage()

        self.prevState = States.Choose

        self.accDeltaAlpha = 0

        self.translateX = 0
        self.translateY = 0

        self.accScale = 0
        self.increase = 1
        self.minScale = 0
        self.maxScale = 123
        self.isScaleInversed = True

        self.warpCnt = 0
        self.warpPointsList = []

        self.fiCntCheck = None
        self.tickCheck = None

    def run(self, handCenterX, handCenterY, fingerCount, fingersLandMarks, state):
        try:
            if state == States.Choose:
                self.fiCntCheck = None
                self.tickCheck = None
                if self.prevState == States.Rotate:
                    self.bigImage = self.getFinalRotatedImage()
                    self.accDeltaAlpha = 0

                if self.prevState == States.Scale:
                    self.bigImage = self.getFinalScaledImage()
                    self.accScale = 0

                self.prevState = States.Choose

            result = self.bigImage.copy()
            if state == States.Translate:
                self.translate(fingersLandMarks[0][0], fingersLandMarks[0][1])

            if state == States.Scale:
                self.scale(fingerCount, fingersLandMarks)

            if state == States.Rotate:
                self.rotate(handCenterX, handCenterY, fingerCount, fingersLandMarks)
                result = self.getFinalRotatedImage()

            if state == States.Brush:
                self.brush(handCenterX, handCenterY, fingerCount, fingersLandMarks)

            if state == States.Warp:
                self.warp(handCenterX, handCenterY, fingerCount, fingersLandMarks)
                # resultImage = self.bigImage.copy()

                # return self.getCroppedImage(resultImage)
                # cv.imshow("Test mask:", self.brushMask[self.Htop: self.Htop + self.h, self.Wtop: self.Wtop + self.w])

            result = self.drawBrushPoints(result)
            return self.getCroppedImage(result)

        except Exception as e:
            print(e)
            logging.error(traceback.format_exc())
            # self.prevState = States.Choose
            # if self.prevState == States.Rotate:
            #     self.bigImage = self.getFinalRotatedImage()
            #     self.accDeltaAlpha = 0

            result = self.drawBrushPoints(self.bigImage.copy())
            return self.getCroppedImage(result)

    def brush(self, handCenterX, handCenterY, fingerCount, fingersLandMarks):
        if fingerCount > 1:
            result = self.fingerCntChecker(fingerCount)
            if result:
                if fingerCount == 5:
                    self.isEraser = True
                else:
                    self.isEraser = False

                    if fingerCount == 2:
                        self.brushColor = RED
                    if fingerCount == 3:
                        self.brushColor = GREEN
                    if fingerCount == 4:
                        self.brushColor = BLUE

            return

        self.fiCntCheck = None
        self.tickCheck = None

        if fingerCount == 0:
            return
        point = fingersLandMarks[0]

        if self.prevState != States.Brush:
            self.prevState = States.Brush
            self.drawCircleX = self.W / 2
            self.drawCircleY = self.H / 2

            self.translateCircleX = point[0]
            self.translateCircleY = point[1]

            self.isEraser = False
            self.isInversedErase = True
            return

        # print(fingerCount)
        # if fingerCount == 5 and self.isInversedErase:
        #     self.isEraser = not self.isEraser
        #     self.isInversedErase = False
        #     # if self.brushColor == RED:
        #     #     self.brushColor = [0, 0, 0]
        #     # else:
        #     #     self.brushColor = RED
        #
        #     return

        # print("YES")
        if fingerCount == 1 or fingerCount == 2:
            self.isInversedErase = True
            deltaX = point[0] - self.translateCircleX
            deltaY = point[1] - self.translateCircleY

            self.translateCircleX = point[0]
            self.translateCircleY = point[1]

            self.drawCircleX += deltaX
            self.drawCircleY += deltaY

            if fingerCount == 1:
                if not self.isEraser:

                    self.drawingList.append((int(self.drawCircleX),
                                             int(self.drawCircleY), self.brushColor), )

                else:
                    eraserPoint = (self.drawCircleX, self.drawCircleY)
                    maxDistance = self.eraserRadius + self.brushRadius
                    self.drawingList = \
                        [point for point in self.drawingList
                         if self.disAB(point, eraserPoint) > maxDistance]
                # cv.circle(self.brushMask, (int(self.drawCircleX),
                #                            int(self.drawCircleY)),
                #           5, self.brushColor, -1)

    def warp(self, handCenterX, handCenterY, fingerCount, fingersLandMarks):
        if fingerCount != 1:
            return

        result = self.fingerCntChecker(fingerCount)

        point = fingersLandMarks[0]
        if result:
            self.warpCnt += 1
            self.warpPointsList.append((int(point[0]), int(point[1])))

            if self.warpCnt == 4:
                self.bigImage[self.Htop: self.Htop + self.h, self.Wtop: self.Wtop + self.w]\
                    = cv.copyMakeBorder(self.bigImage[self.Htop: self.Htop + self.h, self.Wtop: self.Wtop + self.w],
                                                  10,
                                                  10,
                                                  10,
                                                  10,
                                                  cv.BORDER_WRAP)

                self.warpPointsList = []
                self.warpCnt = 0
        pass

    def drawBrushPoints(self, image):
        for data in self.drawingList:
            cv.circle(image, (int(data[0]),
                              int(data[1])),
                      self.brushRadius, data[2], -1)

        return image

    def translate(self, x, y):
        if self.prevState != States.Translate:
            self.prevState = States.Translate
            self.translateX = x
            self.translateY = y
        else:
            height, width = self.bigImage.shape[:2]

            deltaX = x - self.translateX
            deltaY = y - self.translateY

            translation_matrix = np.array(
                [[1, 0, int(deltaX / 5)],
                 [0, 1, int(deltaY / 5)]],
                dtype=np.float32)

            self.bigImage = cv.warpAffine(src=self.bigImage, M=translation_matrix, dsize=(width, height))

    def scale(self, fingerCount, fingersLandMark):
        if fingerCount == 4 and self.isScaleInversed:
            self.increase *= -1
            self.isScaleInversed = False

        if fingerCount != 2:
            return

        self.isScaleInversed = True

        p1 = fingersLandMark[0]
        p2 = fingersLandMark[1]

        distance = self.disAB(p1, p2) / 50
        # print(distance)
        scaleValue = np.interp(distance, [self.minScale, self.maxScale],
                               [0, 1])

        self.accScale += 1 + scaleValue * self.increase

        self.image = cv.resize(self.image, (0, 0),
                               fx=1 + scaleValue * self.increase,
                               fy=1 + scaleValue * self.increase)

        self.bigImage = cv.resize(self.bigImage, (0, 0),
                                  fx=1 + scaleValue * self.increase,
                                  fy=1 + scaleValue * self.increase)

        self.H, self.W = self.bigImage.shape[:2]
        self.h, self.w = self.image.shape[:2]

        self.Htop = int(self.H / 2 - self.h / 2)
        self.Wtop = int(self.W / 2 - self.w / 2)

    def getFinalScaledImage(self):
        return self.bigImage

    def disAB(self, A, B):
        dis = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
        return dis

    def cross(self, a, b):
        return a[0] * b[1] - b[0] * a[1]

    def calcAngle(self, a, b):  # Calculate angle between two vectors
        dot_product = a[0] * b[0] + a[1] * b[1]
        cos_alpha = dot_product / (sqrt(a[0] * a[0] + a[1] * a[1]) * sqrt(b[0] * b[0] + b[1] * b[1]))
        if (cos_alpha > 0.999): cos_alpha = 0.999
        if (cos_alpha < -0.999): cos_alpha = -0.999
        alpha = acos(cos_alpha) * 180 / pi
        if self.cross(a, b - a) < 0:
            alpha *= -1
        return alpha

    def rotate(self, handCenterX, handCenterY, fingerCount, fingerLandMarks):
        vector = None

        if fingerCount > 0:
            vector = np.array([handCenterX, handCenterY]) - np.array(fingerLandMarks[0][0], fingerLandMarks[0][1])

        if self.prevState != States.Rotate or fingerCount != 2:
            self.prevState = States.Rotate

            if vector is not None:
                self.vector = vector
        elif vector is not None:

            deltaAlpha = self.calcAngle(self.vector, vector)

            self.accDeltaAlpha += deltaAlpha
            self.vector = vector

    def getFinalRotatedImage(self):
        height, width = self.bigImage.shape[:2]
        rotation_matrix = cv.getRotationMatrix2D((int(width / 2), int(height / 2)), int(self.accDeltaAlpha), 1.0)

        return cv.warpAffine(self.bigImage, rotation_matrix, (width, height))

    def getCroppedImage(self, image):
        return image[self.Htop: self.Htop + self.h, self.Wtop: self.Wtop + self.w]

    def fingerCntChecker(self, curFingerCnt):
        # print(self.fiCntCheck)

        if self.fiCntCheck is None:
            self.fiCntCheck = curFingerCnt
            self.tickCheck = time.time()
            return False

        if self.fiCntCheck != curFingerCnt:
            self.fiCntCheck = curFingerCnt
            self.tickCheck = time.time()
            return False

        passedTime = time.time() - self.tickCheck

        if passedTime >= 1:
            self.fiCntCheck = None
            self.tickCheck = None
            return True

    def findBigImage(self):
        self.h = self.image.shape[0]
        self.w = self.image.shape[1]

        self.H = self.h * 3
        self.W = self.w * 3
        self.bigImage = np.full((self.H, self.W, 3), 0, np.uint8)

        self.Htop = int(self.H / 2 - self.h / 2)
        self.Wtop = int(self.W / 2 - self.w / 2)

        self.bigImage[self.Htop: self.Htop + self.h, self.Wtop: self.Wtop + self.w] = self.image

        self.brushMask = np.full((self.H, self.W, 3), 0, np.uint8)

        self.drawingList = []
