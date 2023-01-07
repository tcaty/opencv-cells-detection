import numpy as np

from .image import Image


class CellImage(Image):
    def delete_description(self) -> None:
        black_pixel, white_pixel = 0, 255
        description_rows = [
            set([black_pixel, white_pixel]),
            set([black_pixel]),
            set([white_pixel])
        ]
        self._image = np.array(
            list(filter(lambda row: set(row) not in description_rows, self._image))
        )
