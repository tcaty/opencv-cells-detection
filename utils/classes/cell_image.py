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

    def get_cells_using_tresholding(self):
        self.filter(alg=FilterAlg.GAUSSIAN, ksize=(7, 7))
        self.filter(alg=FilterAlg.AVERAGE, ksize=(27, 27))
        self.detect_contours(alg=ContoursDetectingAlg.OTSU_BINARIZATION)
        self.morph_transform(alg=MorphOperation.OPENING,
                             kernel=np.ones((13, 13), np.uint8), iterations=2)
        return self._src

    def get_cells_using_sbd(self):
        # image preprocessing
        self.filter(alg=FilterAlg.GAUSSIAN)
        self.detect_contours(alg=ContoursDetectingAlg.OTSU_BINARIZATION)
        self.morph_transform(alg=MorphOperation.OPENING)
        self.morph_transform(alg=MorphOperation.CLOSING,
                             kernel=np.ones((7, 7), np.uint8), iterations=2)

        contours, _ = cv.findContours(
            self.src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        min_cell_area = 1000  # TODO: detect this one automaticaly
        contours = [contour for contour in contours if cv.contourArea(
            contour) > min_cell_area]

        params = cv.SimpleBlobDetector_Params()
        params.filterByInertia = True
        params.filterByConvexity = True
        params.filterByCircularity = True
        params.filterByArea = False
        params.minInertiaRatio = 0.01
        params.maxInertiaRatio = 0.7
        params.minConvexity = 0.8
        params.minCircularity = 0.1

        detector = cv.SimpleBlobDetector_create(params)
        keypoints = detector.detect(self.src)
        cells_contours = []
        for keypoint in keypoints:
            x, y = tuple(map(int, keypoint.pt))
            for contour in contours:
                if cv.pointPolygonTest(contour, (x, y), False) > 0:
                    cells_contours.append(contour)

        # noiseless mask
        noiseless_mask = np.zeros_like(self.src)
        cv.fillPoly(noiseless_mask, contours, 1)
        self._src = noiseless_mask
        return self.src
