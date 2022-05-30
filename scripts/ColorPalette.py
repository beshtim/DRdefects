import cv2
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

class KMeansReducedPalette:
    """Цветовая палитра
    """
    def __init__(self, num_colors: int, remove_pink: bool):
        """
        Args:
            num_colors (int): Количество цветов в палитре
            remove_pink (bool): Удаляет розовый цвет который заполнятся вне полигона знака
        """
        self.centroid_nearest_pixels = []
        self.num_colors = num_colors
        self.kmeans = MiniBatchKMeans(num_colors, random_state=0xfee1600d, batch_size=256)
        # self.kmeans = KMeans(num_colors, random_state=0xfee1600d)
        self.source_pixels = None
        self.remove_pink = remove_pink

    def _preprocess(self, image):
        if len(image.shape) > 2:
            image = image.reshape(-1, 3)
            if self.remove_pink:
                indexArr = np.argwhere(image == np.array([180, 100, 255]))
                image = np.delete(image, indexArr, 0)
            return image
        print("Error: RGB not found, 3 channels expected")
        return image

    def fit(self, image):
        """Кластеризация пикселей для политры

        Args:
            image (np.ndarray): обрезанная фотография знака
        """
        image_cpy = image.copy()
        self.source_pixels = self._preprocess(image_cpy)
        self.kmeans.fit(self.source_pixels)

        for ci in range(self.num_colors):
            pixels_ci = self.source_pixels[self.kmeans.labels_ == ci]
            distances_ci = np.sqrt(np.sum(np.square(
                pixels_ci - self.kmeans.cluster_centers_[ci]), axis=1))
            pixels_ci = pixels_ci[np.argsort(distances_ci)]
            self.centroid_nearest_pixels.append(pixels_ci)

    def get_palette(self, image: np.ndarray) -> list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.fit(image)
        return self.kmeans.cluster_centers_