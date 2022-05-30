from scripts.GraffitiProcessing import GraffitiProcessing
from scripts.DiscolorProcessing import DiscolorImagesProcessing
from scripts.SurfaceDefectProcessor import SurfaceDefects
from scripts.AnnotationParser import JsonParser
import os
import cv2.cv2 as cv2
import json

class MainProcessor:
    """Класс соединяющий в себе обработку всех типов дефеектов
    """
    def __init__(self, input_imgs, input_jsons, input_csv, df_input, trt, output, cfg):
        self.images_path = input_imgs
        self.df_input = df_input
        self.output = output
        if trt == "PyTorch":
            self.gr_processor = GraffitiProcessing(cfg)
        elif trt == "TensorRT": 
            from scripts.TRTGraffitiProcessor import TRTGraffitiProcessing
            self.gr_processor = TRTGraffitiProcessing(cfg)

        self.dc_processor = DiscolorImagesProcessing(cfg)
        self.sd_processor = SurfaceDefects(cfg)
        self.json_parser = JsonParser(input_jsons, input_csv)

    def process_single_image(self, image_name):
        """Обработка одной картинки

        Args:
            image_name (str): Имя картинки из папки с картинками

        Returns:
            dict: Словарь с результатами
        """
        image_json = self.json_parser.read_json(image_name)
        image_path = os.path.join(self.images_path, image_name)
        image = cv2.imread(image_path)

        result_dc = self.dc_processor.image_process(image, image_json)
        result_sd = self.sd_processor.find_defects(image, image_json)
        result_gr = self.gr_processor.image_process(image, image_json)

        out = {"image_name": image_name,
                "discolor_defect": result_dc,
                "graffiti_defect": result_gr,
                "surface_defect": result_sd}

        json_name = self.df_input[self.df_input["name"] == image_name]["json"].values[0]
        with open(os.path.join(self.output, json_name), "w") as f:
            json.dump(out, f, indent=4)

        return out
