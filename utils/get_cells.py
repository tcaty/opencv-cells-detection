import numpy as np


def get_cells_by_hand(image, threshold_value):
    binary_mask = (image > threshold_value).astype(np.uint8)
    return binary_mask
