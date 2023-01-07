import cv2 as cv
import numpy as np

from utils.get_cells import get_cells_by_hand
from .enums import FilterAlg, MorphOperation, EdgesDetectingAlg, ContoursDetectingAlg


MAP_FILTER_ALG_TO_TUPLE = {
    FilterAlg.AVERAGE: (cv.blur, {'ksize': (10, 10)}),
    FilterAlg.MEDIAN: (cv.medianBlur, {'ksize': 5}),
    FilterAlg.GAUSSIAN: (cv.GaussianBlur, {'ksize': (3, 3), 'sigmaX': 0}),
}

default_kernel = np.ones((5, 5), np.uint8)
MAP_MORP_OPERATION_TO_TUPLE = {
    MorphOperation.EROSION: (cv.erode, {'kernel': default_kernel, 'iterations': 1}),
    MorphOperation.DILATION: (cv.dilate, {'kernel': default_kernel, 'iterations': 1}),
    MorphOperation.OPENING: (cv.morphologyEx, {'op': cv.MORPH_OPEN, 'kernel': default_kernel}),
    MorphOperation.CLOSING: (
        cv.morphologyEx, {'op': cv.morphologyEx, 'kernel': default_kernel})
}

MAP_EDGES_DETECTING_ALG_TO_TUPLE = {
    EdgesDetectingAlg.CANNY: (cv.Canny, {'threshold1': 100, 'threshold2': 200}),
    EdgesDetectingAlg.SOBEL: (
        cv.Sobel, {'ddepth': cv.CV_64F, 'dx': 1, 'dy': 0, 'ksize': 5})
}

MAP_CONTOURS_DETECTING_ALG_TO_TUPLE = {
    ContoursDetectingAlg.BY_HAND: (get_cells_by_hand, {'threshold_value': 100})
}
