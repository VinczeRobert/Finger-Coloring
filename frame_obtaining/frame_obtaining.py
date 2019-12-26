import cv2
import numpy as np

trackbar_name = 'Finger Coloring Application'

class FrameObtainer:
    def __init__(self, cap_region_x_begin, cap_region_y_end, threshold,
                 blur_value, background_sub_threshold, eta):
        self._cap_region_x_begin = cap_region_x_begin
        self._cap_region_y_end = cap_region_y_end
        self._treshold = threshold
        self._blur_value = blur_value
        self._background_sub_threshold = background_sub_threshold
        self._eta = eta

        self._camera = cv2.VideoCapture(0)
        self._camera.set(10, 200)
        self._camera.set(3, 1280)
        self._camera.set(4, 1024)

    def create_trackbar(self):
        cv2.namedWindow(trackbar_name)
        cv2.createTrackbar('thr', trackbar_name, self._treshold, 100, self.print_treshold)

    def print_treshold(thr):
        print("! Changed threshold to " + str(thr))

    def read_frame(self):
        ret, frame = self._camera.read()
        threshold = cv2.getTrackbarPos('thr', trackbar_name)
        # use smoothing filter
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        frame = cv2.flip(frame, 1)
        # flip the frame horizontally
        cv2.rectangle(frame, (int(self._cap_region_x_begin * frame.shape[1]), 0),
                          (frame.shape[1], int(self._cap_region_y_end * frame.shape[0])), (0, 255, 0), 2)
        cv2.imshow('Original Image', frame)
        return frame

    @staticmethod
    def print_treshold(thr):
        print("! Changed threshold to " + str(thr))

    def get_camera(self):
        return self._camera


