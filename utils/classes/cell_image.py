import cv2 as cv
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
        image_with_deleted_descr = np.array(
            list(filter(lambda row: set(row) not in description_rows, self._src))
        )
        self._src = image_with_deleted_descr
        self._original = image_with_deleted_descr

    def get_cells(self):
        self.filter(alg=FilterAlg.GAUSSIAN, ksize=(7, 7))
        self.filter(alg=FilterAlg.AVERAGE, ksize=(13, 13))
        self.detect_contours(alg=ContoursDetectingAlg.OTSU_BINARIZATION)
        contours, _ = cv.findContours(
            self.src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = [
            contour for contour in contours if cv.contourArea(contour) > 1500]

        filtered_contours = []
        deltas = []
        for contour in contours:
            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            delta = cv.arcLength(contour, True) - cv.arcLength(approx, True)
            deltas.append(delta)
            if cv.contourArea(contour) > 1500 and delta < 100:
                filtered_contours.append(contour)

        self.transform(lambda src: np.full_like(src, 0))
        cv.fillPoly(self.src, filtered_contours, 255)

        return self.src, deltas
