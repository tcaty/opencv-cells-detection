import cv2 as cv
import numpy as np

from utils.helpers.helpers import create_algs_wrapper
from utils.helpers.enums import FilterAlg, MorphOperation, EdgesDetectingAlg, ContoursDetectingAlg
from utils.helpers.mappings import MAP_FILTER_ALG_TO_TUPLE, MAP_EDGES_DETECTING_ALG_TO_TUPLE, MAP_MORP_OPERATION_TO_TUPLE, MAP_CONTOURS_DETECTING_ALG_TO_TUPLE


class Image:
    def __init__(self, src: cv.Mat) -> None:
        self._src = src
        self._meta = ()  # store for last actions additional data

    def __alg(map_alg_name_to_tuple):  # type: ignore
        def decorator(method):
            def wrapped(self, alg, overwrite=True, **params):
                method(self, alg, overwrite)
                alg_wrapper = create_algs_wrapper(
                    map_alg_name_to_tuple)
                temp_result = alg_wrapper(self._src, alg, **params)
                result_is_tuple = isinstance(temp_result, tuple)
                alg_result = temp_result if not result_is_tuple else temp_result[-1]
                self._meta = temp_result[:-1]
                if (overwrite):
                    self._src = alg_result
                return alg_result
            return wrapped
        return decorator

    @__alg(MAP_FILTER_ALG_TO_TUPLE)  # type: ignore
    def filter(self, alg: FilterAlg, overwrite=True, **params) -> None:
        pass

    @__alg(MAP_MORP_OPERATION_TO_TUPLE)  # type: ignore
    def morph_transform(self, alg: MorphOperation, overwrite=True, **params) -> None:
        pass

    @__alg(MAP_EDGES_DETECTING_ALG_TO_TUPLE)  # type: ignore
    def detect_edges(self, alg: EdgesDetectingAlg, overwrite=True, **params) -> None:
        pass

    @__alg(MAP_CONTOURS_DETECTING_ALG_TO_TUPLE)  # type: ignore
    def detect_contours(self, alg: ContoursDetectingAlg, overwrite=True, **params) -> None:
        pass

    def transform(self, trans):
        transformed = trans(self._src)
        self._src = transformed
        return transformed

    @property
    def src(self) -> cv.Mat:
        return self._src

    @property
    def meta(self) -> tuple:
        return self._meta
