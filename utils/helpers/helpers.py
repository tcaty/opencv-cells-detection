import os
import cv2
import matplotlib.pyplot as plt


def get_images(dir_path):
    file_names = os.listdir(dir_path)
    images_file_names = list(
        filter(lambda file_name: '.txt' not in file_name, file_names))
    return [cv2.imread(f'{dir_path}/{image_file_name}', cv2.COLOR_BGR2GRAY) for image_file_name in images_file_names]


def show_images(image_with_title_tuples, grid=(2, 2), size_inches=(10, 10)):
    plt.gcf().set_size_inches(*size_inches)
    for i in range(len(image_with_title_tuples)):
        current_tuple = image_with_title_tuples[i]
        plt.subplot(*grid, i + 1)
        plt.imshow(current_tuple[0], 'gray')
        plt.title(current_tuple[1])
        plt.xticks([])
        plt.yticks([])


def create_algs_wrapper(map_alg_name_to_tuple):
    def algs_wrapper(image, alg, **params):
        algs_dict = map_alg_name_to_tuple[alg]
        current_alg, default_params = algs_dict[0], algs_dict[1]
        default_params.update(params)
        transformed_images = current_alg(image, **default_params)
        return transformed_images
    return algs_wrapper
