import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_cdt

class DistanceTransformCalculator:

    @staticmethod
    def calculate_distance_transform(source_image):
        edt = distance_transform_cdt(source_image, metric='taxicab')
        dt = np.uint8(edt)
        cv2.imshow('Distance Transform', dt)
        return dt