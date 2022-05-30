from scipy.spatial import distance
import numpy as np
import cv2 as cv2
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import pickle
from scripts.ColorPalette import KMeansReducedPalette

class DiscolorImagesProcessing:
    """Класс реализующий определение выцветания
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        orig_palettes_pkl_path = cfg["orig_palettes_pkl_path"]
        k_colors = 6
        remove_pink = True
        self.KMPalette = KMeansReducedPalette(k_colors, remove_pink)
        self.from_int_to_type = {0: "normal", 1: "colored", -1: "No orig palette for this class {}"}
        with open(orig_palettes_pkl_path, 'rb') as f:
            self.all_orig_palettes = pickle.load(f)

    def image_process(self, image: np.ndarray, image_json):
        """Проверка одного знака

        Args:
            image (np.ndarray): Фотография со знаком
            image_json (dict): json картинки

        Returns:
            str: True/Falce - есть дефект или нет
        """
        image_class = image_json["class"]
        masked_im_discolor = self.masked_image_from_polygon(image, image_json["segmentation"][0], image_json["bbox"][0])
        palette_discolor = self.KMPalette.get_palette(masked_im_discolor)

        # ---
        if image_class in self.all_orig_palettes.keys():
            orig_palette = self.all_orig_palettes[image_class]
        else:
            return None

        distance = self.mean_distance(orig_palette, palette_discolor, "cie2000")
        threshold = self.cfg["discolor_threshold"]
        if distance > threshold:
            return str(True)
        else:
            return str(False)

    def NN(self, A, start):
        """Nearest neighbor algorithm 
        """
        path = [start]
        cost = 0
        N = A.shape[0]
        mask = np.ones(N, dtype=bool)
        mask[start] = False

        for i in range(N - 1):
            last = path[-1]
            next_ind = np.argmin(A[last][mask])
            next_loc = np.arange(N)[mask][next_ind]
            path.append(next_loc)
            mask[next_loc] = False
            cost += A[last, next_loc]

        return path, cost

    def sort_colors(self, palette, metric: str) -> list:
        """Сортировка палитры

        Args:
            palette (np.ndarray): Палитра знака
            metric (str): Используется CIE2000 

        Returns:
            list: Список отсортированных цветов
        """
        if metric == "contrast":
            palette_gray = palette @ np.array([[0.21, 0.71, 0.07]]).T
            idx = palette_gray.flatten().argsort()
            palette_sorted = palette[idx]
            return palette_sorted

        colours_length = len(palette)
        A = np.zeros([colours_length, colours_length])
        for x in range(0, colours_length - 1):
            for y in range(0, colours_length - 1):
                if metric == "euclidean":
                    A[x, y] = distance.euclidean(palette[x], palette[y])
                if metric == "cie2000":
                    color1_lab = convert_color(sRGBColor(*palette[x]), LabColor)
                    color2_lab = convert_color(sRGBColor(*palette[y]), LabColor)
                    A[x, y] = delta_e_cie2000(color1_lab, color2_lab)

        path, _ = self.NN(A, 0)

        colours_nn = []
        for i in path:
            colours_nn.append(palette[i])
        return colours_nn

    def masked_image_from_polygon(self, image: np.ndarray, segment: list, bbox: list) -> np.ndarray:
        """Обрезание знака по полигону и подстановка маски для пикселей не принадлежащих полигону

        Args:
            image (np.ndarray): Фото картинки
            segment (list): массив из точек (x1,y1 ... xi,yi)
            bbox (list): бибокс картинки (x,y,h,w)

        Returns:
            np.ndarray: Итоговая картинка знака
        """
        mask = np.zeros(image.shape, dtype=np.uint8)
        polygon = np.array([[segment[i:i + 2] for i in range(0, len(segment), 2)]], dtype=np.int32)
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, polygon, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        mask = mask[int(bbox[1]):int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2])]
        masked_image = masked_image[int(bbox[1]):int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2])]
        masked_image[(mask == 0).all(-1)] = [180, 100, 255]
        return masked_image

    def mean_distance(self, palette_orig, palette_discolor, metric) -> float:
        """Среднее расстояние между палитрами

        Args:
            palette_orig (np.ndarray): Оригинальная палитра для этого типа знака
            palette_discolor (np.ndarray): Полученная палитра знака
            metric (str): Метрика - cie2000

        Returns:
            float: _description_
        """
        if metric == "cie2000":
            mean_orig = np.mean(palette_orig, axis=0)
            mean_discolor = np.mean(palette_discolor, axis=0)

            color1_lab = convert_color(sRGBColor(*mean_orig), LabColor)
            color2_lab = convert_color(sRGBColor(*mean_discolor), LabColor)
            delta_e = delta_e_cie2000(color1_lab, color2_lab)
            return delta_e