import logging
import traceback

import cv2 as cv
import numpy as np
import threading
import keyboard
import math


class HandRecognition:
    def __init__(self):
        # BOUNDING_RECT_NEIGHBOR_DISTANCE_SCALING
        self.BRNebDis_SCALING = 0.05
        self.closedHandArea = 0

    def run(self, handPixels, originalFrame, handContour):
        frame = np.copy(originalFrame)
        fingerCount = 0
        fingersLandMarks = []

        try:
            epsilon = 0.0005 * cv.arcLength(handContour, True)
            approx = cv.approxPolyDP(handContour, epsilon, True)

            hull = cv.convexHull(handContour)
            copyHull = hull.copy()
            # cv.drawContours(frame, [hull], -1, (0, 255, 0), 2)
            # cv.drawContours(frame, approx, -1, (0, 255, 0), 2)

            h, w, cx, cy, frame = self.boundingRectangle(hull, frame.copy(),
                                                         drawRec=False,
                                                         drawHandCenter=False)
            # x =
            areaHull = cv.contourArea(hull)
            areaHandContour = cv.contourArea(handContour)
            # find the percentage of area not covered by hand in convex hull
            # areaRatio = ((areaHull - areaHandContour) / areaHull) * 100
            # print("HW:", h*w, "area:", areaHandContour, "min:", h*w - areaHandContour, "dividee:", )
            areaRatio = ((areaHull - areaHandContour) / areaHandContour) * 100

            '''
            pointsList = self.convertHullToList(hull)
            pointsList = self.compactOnMedian(pointsList, h * self.BRNebDis_SCALING)

            pointsList = self.compactOnAxis(pointsList, axis=0, threshold=20)
            pointsList = self.compactOnAxis(pointsList, axis=1, threshold=20)

            for i in range(len(pointsList)):
                x = pointsList[i][0]
                y = pointsList[i][1]
                if y < cy - 5:
                    fingersLandMarks.append((x, y))
                    fingerCount += 1

            print(areaRatio)
            # if areaRatio < 30:
            #     fingersLandMarks = []
            #     fingerCount = 0

            for landMark in fingersLandMarks:
                cv.circle(frame, (int(landMark[0]), int(landMark[1])), 5, [0, 0, 255], -1)

            '''

            hull = cv.convexHull(approx, returnPoints=False)
            defects = cv.convexityDefects(approx, hull)
            lastFinger = None
            test = set()

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])

                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                # distance between point and convex hull
                d = (2 * ar) / a

                # apply cosine rule here
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if abs(angle) < 90 and d > 30 and end[1] < cy - 5 and start[1] < cy - 5:
                    fingerCount += 1
                    fingersLandMarks.append(end)
                    test.add(end)
                    test.add(start)

                    cv.circle(frame, end, 5, [0, 0, 255], -1)
                    cv.line(frame, start, far, [0, 255, 0], 2)
                    cv.line(frame, far, end, [0, 255, 0], 2)

                # cv.line(frame, start, end, [0, 255, 0], 2)

            fingerCount += 1
            font = cv.FONT_HERSHEY_SIMPLEX

            if len(test) > 1:
                result = (0, 0)
                for point in test:
                    if result[0] < point[0]:
                        result = point

                lastFinger = result

            if lastFinger is not None:
                cv.circle(frame, lastFinger, 5, [0, 0, 255], -1)
                fingersLandMarks.append(lastFinger)

            if fingerCount == 1:
                if areaHandContour < 2000:
                    fingerCount = 0
                    fingersLandMarks = []
                    cv.putText(frame, 'Put hand in the box', (0, 50), font, 1, (255, 0, 0), 2, cv.LINE_AA)
                else:
                    pointsList = self.convertHullToList(copyHull)
                    res = (0, 0)
                    mx = 0
                    for point in pointsList:
                        dis = self.disAB(point, (cx, cy))
                        if point[1] < cy and dis > mx:
                            res = point
                            mx = dis

                    if mx > 130:
                        fingersLandMarks.append(res)
                        cv.circle(frame, res, 5, [0, 0, 255], -1)
                        cv.putText(frame, '1', (0, 50), font, 2, (255, 0, 0), 2, cv.LINE_AA)
                    else:
                        fingerCount = 0
                        cv.putText(frame, '0', (0, 50), font, 2, (255, 0, 0), 2, cv.LINE_AA)

            elif fingerCount == 2:
                cv.putText(frame, '2', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)

            elif fingerCount == 3:
                cv.putText(frame, '3', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)

            elif fingerCount == 4:
                cv.putText(frame, '4', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)

            elif fingerCount == 5:
                cv.putText(frame, '5', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)

            elif fingerCount == 6:
                pass

            return frame, cx, cy, fingerCount, fingersLandMarks

        except Exception as e:
            # logging.error(traceback.format_exc())
            return frame, None, None, None, None
            pass

    def compactOnMedian(self, points, maxNebDistance):
        ref = None
        median = None

        medianPoints = []
        for i in range(len(points)):
            cur = (points[i][0], points[i][1])

            if i == 0:
                ref = cur
                median = cur
            else:
                if self.disAB(ref, cur) > maxNebDistance:

                    medianPoints.append(median)
                    ref = cur
                    median = cur
                else:
                    midX = (cur[0] + median[0]) / 2
                    midY = (cur[1] + median[1]) / 2

                    median = (midX, midY)

        return medianPoints

    def disAB(self, A, B):
        dis = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
        return dis

    def boundingRectangle(self, hull, frame, drawRec=True, drawHandCenter=True):
        x, y, w, h = cv.boundingRect(hull)
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        if drawRec:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if drawHandCenter:
            cv.circle(frame, (int(cx), int(cy)), 5, [0, 0, 255], -1)

        return h, w, cx, cy, frame

    def compactOnAxis(self, points, axis, threshold):
        sor = points.copy()
        sor.sort(key=lambda x: x[axis])

        result = []
        ref = None

        for i in range(len(sor)):
            if i == 0:
                ref = sor[0]
            else:
                if abs(ref[axis] - sor[i][axis]) < threshold:
                    if ref[1 - axis] > sor[i][1 - axis]:
                        ref = sor[i]
                else:
                    result.append(ref)
                    ref = sor[i]

        result.append(ref)
        return result

    def convertHullToList(self, hull):
        li = []
        for i in range(hull.shape[0]):
            x = hull[i][0][0]
            y = hull[i][0][1]
            cur = (x, y)
            li.append(cur)

        return li
