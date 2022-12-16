import cv2 as cv
import numpy as np
import threading
import keyboard
import math


class HandRecognition:
    def __init__(self):
        pass

    def run(self, handPixels, originalFrame, handContour):
        frame = np.copy(originalFrame)

        # print(handContour.shape)
        if handContour is None or len(handContour) == 0:
            return

        try:  # an error comes if it does not find anything in window as it cannot find contour of max area
            # therefore this try error statement
            kernel = np.ones((3, 3), np.uint8)

            epsilon = 0.0005 * cv.arcLength(handContour, True)
            approx = cv.approxPolyDP(handContour, epsilon, True)

            hull = cv.convexHull(handContour)

            areaHull = cv.contourArea(hull)
            areaHandContour = cv.contourArea(handContour)

            # find the percentage of area not covered by hand in convex hull
            areaRatio = ((areaHull - areaHandContour) / areaHandContour) * 100

            # find the defects in convex hull with respect to hand
            hull = cv.convexHull(approx, returnPoints=False)
            defects = cv.convexityDefects(approx, hull)

            # l = no. of defects
            l = 0

            # code for finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt = (100, 180)

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
                if angle <= 90 and d > 30:
                    l += 1
                    cv.circle(frame, far, 3, [0, 0, 255], -1)

                # draw lines around hand
                cv.line(frame, start, end, [0, 255, 0], 2)

            l += 1

            # print corresponding gestures which are in their ranges
            font = cv.FONT_HERSHEY_SIMPLEX
            if l == 1:

                if areaHandContour < 2000:
                    pass
                    cv.putText(frame, 'Put hand in the box', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)
                else:
                    cv.putText(frame, '1', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)

                    # if areaRatio < 10:
                    #     cv.putText(frame, 'fist vertical', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)
                    # elif areaRatio < 18:
                    #     cv.putText(frame, 'thumbs down', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)
                    # elif areaRatio < 27:
                    #     cv.putText(frame, 'Best of luck', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)
                    # else:
                    #     cv.putText(frame, '1', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)


            elif l == 2:
                # if areaRatio < 40:
                cv.putText(frame, '2', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)
                # else:
                #     pass
                    # cv.putText(frame, 'L', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)

            elif l == 3:
                # if areaRatio < 27:
                cv.putText(frame, '3', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)
                # else:
                #     pass
                    # cv.putText(frame, 'ok', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)

            elif l == 4:
                cv.putText(frame, '4', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)

            elif l == 5:
                cv.putText(frame, '5', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)

            elif l == 6:
                pass
                # cv.putText(frame, 'reposition', (0, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)

            else:
                pass
                # cv.putText(frame, 'reposition', (10, 50), font, 2, (255, 0, 0), 3, cv.LINE_AA)

            cv.imshow('frame', frame)
        except:
            pass

    def boundingRectangle(self):
        pass
