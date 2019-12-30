import cv2


def convert_to_binary(color_image, blur_value, threshold):
    gray_image =cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blur_value, blur_value), 0)
    _, binary_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Binary Image', binary_image)
    return binary_image