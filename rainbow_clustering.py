from typing import Tuple, Union

import cv2
import numpy as np
from sklearn.cluster import KMeans


class RainbowClustering:
    def __init__(
            self,
            clusters: int,
            feature_mode: str = 'hsv',
            verbose: bool = False,
            white_threshold: int = 250,
            black_threshold: int = 10,
            clustering_method: str = 'kmeans'
    ) -> None:
        """
        Class for clustering different rainbow images. White and black background will be binarized as single
        background class. Set white_threshold to 256 and/or black_threshold to -1 to exclude binarization.

        :param clusters: number of clusters (background is not included here).
        :param feature_mode: use 'hsv' or 'rgb' features for clustering.
        :param verbose: set to True for verbosity mode.
        :param white_threshold: threshold value for binarization of the white color.
        :param black_threshold: threshold value for binarization of the black color.
        :param clustering_method: clustering method, only 'kmeans' is available now.
        """
        self.clusters = clusters + 1
        self.verbose = verbose

        self.feature_mode = feature_mode
        if self.feature_mode == 'hsv':
            self.feature_mode_factor = np.array([179, 255, 255])
        elif self.feature_mode == 'rgb':
            self.feature_mode_factor = np.array([255, 255, 255])
        else:
            raise ValueError(f'Available feature mode are ("hsv", "rgb"), got "{feature_mode}".')

        if not 0 <= white_threshold <= 255:
            print(f'White threshold value is out of borders (got "{white_threshold}"). Setting to "256".')
            white_threshold = 256
        if not 0 <= black_threshold <= 255:
            print(f'Black threshold value is out of borders (got "{black_threshold}"). Setting to "-1".')
            black_threshold = -1
        self.white_threshold = white_threshold
        self.black_threshold = black_threshold

        if clustering_method == 'kmeans':
            self.clustering = KMeans(n_clusters=self.clusters, verbose=verbose)
        else:
            raise ValueError(f'Available clustering_method is "kmeans", got "{clustering_method}".')

    def _prepare_feature_vectors(self, image: np.ndarray, grid_weight: Union[str, float] = 'default') -> np.ndarray:
        """
        Obtaining feature vectors based on the input image.

        :param image: RGB image numpy array with shape (height, width, 3).
        :param grid_weight: weight for pixels coordinates features.
            "default" uses clusters number as weight value; set 0 to exclude grid features or any other float value.
        :return: feature vectors array with shape (height * width, features number).
        """
        gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)

        # Thresholding the white and black colors.
        white_mask = np.where(gray_image >= self.white_threshold, 1, 0)
        black_mask = np.where(gray_image <= self.black_threshold, 1, 0)
        white_black_mask = np.expand_dims(np.logical_or(white_mask, black_mask), axis=2)
        thresh_image = np.where(white_black_mask, 0, image)
        if np.sum(white_black_mask) == 0:
            print('The background was not found. There will be no background cluster.')
            self.clusters -= 1
            self.clustering.n_clusters = self.clusters

        # Converting image to chosen feature mode and normalizing it.
        if self.feature_mode == 'hsv':
            feature_image = cv2.cvtColor(thresh_image, cv2.COLOR_RGB2HSV)
        elif self.feature_mode == 'rgb':
            feature_image = thresh_image
        else:
            raise ValueError(f'Available feature mode are ("hsv", "rgb"), got "{self.feature_mode}".')
        feature_image = feature_image / np.expand_dims(self.feature_mode_factor, axis=(0, 1))

        # Add pixels coordinates as features.
        grid = np.meshgrid(np.arange(feature_image.shape[1]), np.arange(feature_image.shape[0]))
        grid = np.stack(grid, axis=2)
        # Make background vectors far from non-background ones.
        grid = np.where(white_black_mask, -1000, grid)
        # Normalizing and adding weights to it. It allows you to obtain clusters with similar size and low noise.
        if grid_weight == 'default':
            grid_weight = self.clusters
        elif isinstance(grid_weight, str):
            raise ValueError(f'Expected grid_weight to be "default" or any float value, got "{grid_weight}".')
        grid = grid / np.max(grid) * grid_weight

        # Create feature vectors.
        feature_vectors = np.reshape(
            np.concatenate([feature_image, grid], axis=2),
            (feature_image.shape[0] * feature_image.shape[1], feature_image.shape[2] + grid.shape[2])
        )
        return feature_vectors

    def predict(self, image: np.ndarray, grid_weight: Union[str, float] = 'default') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make pixel clustering of single RGB image.

        :param image: RGB image numpy array with shape (height, width, 3).
        :param grid_weight: weight for pixels coordinates features.
            "default" uses clusters number as weight value; set 0 to exclude grid features or any other float value.
        :return: mask numpy array with shape (height, width) and centroids numpy array with shape (clusters number, 3).
         - Mask values are indexes of the corresponding centroids vectors, e.g. 4 corresponds to centroids[4, :].
         - Every centroid vector is an RGB color vector.
        """
        vectors = self._prepare_feature_vectors(image, grid_weight)

        clustering_result = self.clustering.fit_predict(vectors)
        centroids = self.clustering.cluster_centers_[:, :3] * self.feature_mode_factor
        centroids = centroids.astype(np.uint8)
        if self.feature_mode == 'hsv':
            centroids = cv2.cvtColor(np.expand_dims(centroids, 0), cv2.COLOR_HSV2RGB)[0, :, :]

        mask = np.reshape(clustering_result, (image.shape[0], image.shape[1]))
        return mask, centroids
