import cv2 as cv
import numpy as np

from utils.helpers.helpers import create_algs_wrapper
from utils.helpers.enums import FilterAlg, MorphOperation, EdgesDetectingAlg, ContoursDetectingAlg
from utils.helpers.mappings import MAP_FILTER_ALG_TO_TUPLE, MAP_EDGES_DETECTING_ALG_TO_TUPLE, MAP_MORP_OPERATION_TO_TUPLE, MAP_CONTOURS_DETECTING_ALG_TO_TUPLE


class Image:
    def __init__(self, image: cv.Mat) -> None:
        self._image = image

    def __alg(map_alg_name_to_tuple):  # type: ignore
        def decorator(method):
            def wrapped(self, alg_name, overwrite_image=True, **params):
                method(self, alg_name, overwrite_image)
                alg_wrapper = create_algs_wrapper(
                    map_alg_name_to_tuple)
                alg_result = alg_wrapper(self._image, alg_name, **params)
                if (overwrite_image):
                    self._image = alg_result
                return alg_result
            return wrapped
        return decorator

    @__alg(MAP_FILTER_ALG_TO_TUPLE)  # type: ignore
    def filter(self, alg_name: FilterAlg, overwrite_image=True, **params) -> None:
        pass

    @__alg(MAP_MORP_OPERATION_TO_TUPLE)  # type: ignore
    def morph_transform(self, alg_name: MorphOperation, overwrite_image=True, **params) -> None:
        pass

    @__alg(MAP_EDGES_DETECTING_ALG_TO_TUPLE)  # type: ignore
    def detect_edges(self, alg_name: EdgesDetectingAlg, overwrite_image=True, **params) -> None:
        pass

    @__alg(MAP_CONTOURS_DETECTING_ALG_TO_TUPLE)  # type: ignore
    def detect_contours(self, alg_name: ContoursDetectingAlg, overwrite_image=True, **params) -> None:
        pass

    @property
    def image(self) -> cv.Mat:
        return self._image
