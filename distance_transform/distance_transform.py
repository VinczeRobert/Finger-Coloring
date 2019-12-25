import cv2
import numpy as np

class DistanceTransformCalculator:

    @staticmethod
    def calculate_distance_transform(source_image, type, mask_size):
        h, w = source_image.shape[:2]
        dst = np.zeros((h, 2), np.float32)
        cv2.distanceTransform(source_image, type, mask_size, dst=dst)
        dt = np.array(np.uint8(dst))
        cv2.imshow('Distance Transform', dt)
        return dt