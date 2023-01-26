import time

from States import States
import cv2 as cv

class StateProcessor:
    def __init__(self):
        self.state = States.Choose

        self.fiCntCheck = None
        self.tickCheck = None
        self.font = cv.FONT_HERSHEY_SIMPLEX

    def run(self, fingerCount, frame):
        if fingerCount is None or fingerCount >= 6:
            self.reset()

        elif self.state == States.Choose:
            self.chooseStateProcess(fingerCount)

        else:
            if fingerCount == 0:
                if self.fingerCntChecker(fingerCount):
                    self.state = States.Choose
            else:
                self.resetFingerCntChecker()

        '''
        elif self.state == States.Translate:
            self.translateStateProcess(fingerCount)

        elif self.state == States.Scale:
            self.scaleStateProcess(fingerCount)
        '''
        cv.putText(frame, self.state.name, (120, 200), self.font, 1, (255, 0, 0), 2, cv.LINE_AA)

        return frame

    def chooseStateProcess(self, fingerCount):
        if self.fingerCntChecker(fingerCount):
            self.state = States(fingerCount)

    def translateStateProcess(self, fingerCount):
        if fingerCount == 0:
            if self.fingerCntChecker(fingerCount):
                self.state = States.Choose
        else:
            self.resetFingerCntChecker()

    def scaleStateProcess(self, fingerCount):
        pass

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

    def reset(self):
        self.state = States.Choose
        self.resetFingerCntChecker()

    def resetFingerCntChecker(self):
        self.fiCntCheck = None
        self.tickCheck = None
