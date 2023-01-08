import numpy as np

from .image import Image
from utils.helpers.enums import Pixel


class CellImage(Image):
    def delete_description(self) -> None:
        description_rows = [
            set([Pixel.WHITE_GRAY.value, Pixel.BLACK_GRAY.value]),
            set([Pixel.BLACK_GRAY.value]),
            set([Pixel.WHITE_GRAY.value])
        ]
        self._src = np.array(
            list(filter(lambda row: set(row) not in description_rows, self._src))
        )
