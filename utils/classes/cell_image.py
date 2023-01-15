import numpy as np

from .image import Image
from utils.helpers.enums import Pixel, FilterAlg, MorphOperation, ContoursDetectingAlg, EdgesDetectingAlg


class CellImage(Image):
    def delete_description(self):
        description_rows = [
            set([Pixel.WHITE_GRAY.value, Pixel.BLACK_GRAY.value]),
            set([Pixel.BLACK_GRAY.value]),
            set([Pixel.WHITE_GRAY.value])
        ]
        self._src = np.array(
            list(filter(lambda row: set(row) not in description_rows, self._src))
        )

    def get_cells(self):
        self.filter(alg=FilterAlg.GAUSSIAN, ksize=(7, 7))
        self.filter(alg=FilterAlg.AVERAGE, ksize=(27, 27))
        self.detect_contours(alg=ContoursDetectingAlg.OTSU_BINARIZATION)
        self.morph_transform(alg=MorphOperation.OPENING,
                             kernel=np.ones((13, 13), np.uint8), iterations=2)
        return self.src
