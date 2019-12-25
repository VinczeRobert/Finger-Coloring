import cv2
import numpy as np

class BackgroundSubtractor:

    def __init__(self, background_sub_threshold, eta):
        self._frame = None
        self._background_model = cv2.createBackgroundSubtractorMOG2(0, background_sub_threshold)
        self._eta = eta
        self._background_captured = False

    def extract_background(self, cap_region_y_end, cap_region_x_begin):
        mask = self._background_model.apply(self._frame, learningRate=self._eta)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        # get the background by checking the difference
        img = cv2.bitwise_and(self._frame, self._frame, mask=mask)
        img = img[0:int(cap_region_y_end * self._frame.shape[0]),
              int(cap_region_x_begin * self._frame.shape[1]):self._frame.shape[1]]
        cv2.imshow('Extracted Hand', self._frame)
        return img

    def set_frame(self, frame):
        self._frame = frame

    def is_background_captured(self):
        return self._background_captured

    def set_background_captured(self, background_captured):
        self._background_captured = background_captured

