import cv2

class BinaryConvertor:

    def __init__(self, color_image):
        self._color_image = color_image

    def convert_to_binary(self, blur_value, threshold):
        gray_image =cv2.cvtColor(self._color_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (blur_value, blur_value), 0)
        _, binary_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('Binary Image', binary_image)
        return binary_image