import cv2
import numpy as np

class PalmSegmentor():

    def __init__(self):
        self._max_i = -1
        self._max_j = -1

    def obtaining_palm_point(self, dt):
        self._max_i, self._max_j = np.unravel_index(np.argmax(dt, axis=None), dt.shape)
        return self._max_i, self._max_j

    @staticmethod
    def from_one_channel_to_three(original_image, binary_image):
        three_channels = np.zeros_like(3 * original_image)
        three_channels[:,:,0] = binary_image
        three_channels[:,:,1] = binary_image
        three_channels[:,:,2] = binary_image

        return three_channels

    def draw_image_with_palm_point(self, image):
        cv2.circle(image, (self._max_j, self._max_i), 5, (0, 255, 0), -1)
        cv2.imshow('Picture with Palm Point', image)