from enum import Enum


class Pixel(Enum):
    WHITE_GRAY = 255
    BLACK_GRAY = 0
    WHITE_RGB = [255, 255, 255]
    BLACK_RGB = [0, 0, 0]
    RED = [255, 0, 0]
    GREEN = [0, 255, 0]
    BLUE = [0, 0, 255]


class FilterAlg(Enum):
    GAUSSIAN = 0
    MEDIAN = 1
    AVERAGE = 2


class MorphOperation(Enum):
    EROSION = 0
    DILATION = 1
    OPENING = 2
    CLOSING = 3


class EdgesDetectingAlg(Enum):
    SOBEL = 0
    CANNY = 1


class ContoursDetectingAlg(Enum):
    BY_HAND = 0
