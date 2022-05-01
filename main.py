import os
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from tqdm import tqdm

from rainbow_clustering import RainbowClustering


def create_color_mask(
        mask_shape: Tuple[int, int, int], clustering_mask: np.ndarray, centroids_array: np.ndarray
) -> np.ndarray:
    """
    Create color mask of clustering results.

    :param mask_shape: shape of original RGB image (height, width, 3).
    :param clustering_mask: clustering mask array with shape (clusters number, 3)
    :param centroids_array: centroids numpy array from clustering.
    :return: color mask array as an RGB image with shape (height, width, 3).
    """
    mask = np.zeros(mask_shape, dtype=np.uint8)
    clustering_mask = np.expand_dims(clustering_mask, axis=2)
    centroids_array = np.expand_dims(centroids_array, axis=(0, 1))
    for _i in range(centroids_array.shape[2]):
        mask = np.where(clustering_mask == _i, centroids_array[:, :, _i, :], mask)
    return mask


def process_image(
        img_path: str, save_path: str, feature_mode: str = 'rgb', grid_weight: Union[str, float] = 'default'
) -> None:
    """
    Process an image with different amount of clusters, plot all masks together and save plots to a file.

    :param img_path: path to image to be processed.
    :param save_path: path to existed folder to save plots file.
    :param feature_mode: use 'hsv' or 'rgb' features for clustering.
    :param grid_weight: weight for pixels coordinates features.
            "default" uses clusters number as weight value; set 0 to exclude grid features or any other float value.
    """
    clusters_list = [1, 2, 3, 4, 7, 12, 20]

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10), dpi=200, sharex='all', sharey='all')
    fig.suptitle(f'Image "{os.path.basename(img_path)}", feature_mode "{feature_mode}", grid_weight "{grid_weight}".')
    axes[0, 0].imshow(image)
    axes[0, 0].set(title='Original image')
    axes[0, 0].axis('off')

    for i, clusters in enumerate(tqdm(clusters_list, desc='Cluster values', leave=False), start=1):
        clustering_obj = RainbowClustering(clusters=clusters, feature_mode=feature_mode)
        result, centroids = clustering_obj.predict(image, grid_weight=grid_weight)
        color_mask = create_color_mask(image.shape, result, centroids)

        axes[i // 4, i % 4].imshow(color_mask)
        axes[i // 4, i % 4].set(title=f'{clusters} clusters')
        axes[i // 4, i % 4].axis('off')

    fig.savefig(os.path.join(save_path, os.path.basename(img_path)), bbox_inches='tight')
    plt.close(fig)
    plt.close('all')


if __name__ == '__main__':
    IMAGES_PATH = 'images'
    save_path_rgb = 'results_rgb'
    save_path_hsv = 'results_hsv'
    os.makedirs(save_path_rgb, exist_ok=True)
    os.makedirs(save_path_hsv, exist_ok=True)

    rcParams.update({'font.size': 18})

    images_with_grid = ['rainbow.png', 'rainbow_2.png', 'rainbow_3.png', 'rainbow_4.png']
    images_without_grid = ['rainbow_5.png', 'spectrum_1.jpg', 'spectrum_2.jpg']
    spectrum = [os.path.join(IMAGES_PATH, p) for p in os.listdir(IMAGES_PATH) if 'spectrum' in p]

    for ft_mode, s_path in zip(['rgb', 'hsv'], [save_path_rgb, save_path_hsv]):
        for path in tqdm(images_with_grid, desc=f'Rainbow images, {ft_mode} mode'):
            process_image(
                img_path=os.path.join(IMAGES_PATH, path),
                save_path=s_path,
                feature_mode=ft_mode,
                grid_weight='default'
            )
        for path in tqdm(images_without_grid, desc=f'Spectrum images, {ft_mode} mode'):
            process_image(
                img_path=os.path.join(IMAGES_PATH, path),
                save_path=s_path,
                feature_mode=ft_mode,
                grid_weight=0
            )
