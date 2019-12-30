import cv2
import numpy as np


def create_components(image):
    # ensure binary
    img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    ret, labels = cv2.connectedComponents(img)

    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('Labeled', labeled_img)
    return labeled_img