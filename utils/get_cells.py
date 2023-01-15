import cv2 as cv
import numpy as np


def get_cells_by_hand(image: cv.Mat, threshold_value, backgroundIsDarker=True):
    binary_mask = image > threshold_value if backgroundIsDarker else image < threshold_value
    return binary_mask.astype(np.uint8)
