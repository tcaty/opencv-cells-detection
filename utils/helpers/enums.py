from enum import Enum


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
