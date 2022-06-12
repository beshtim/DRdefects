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
    def __init__(self, input_imgs, input_jsons, trt, output, cfg):
        self.images_path = input_imgs
        self.input_jsons = input_jsons
        self.output = output
        if trt == "PyTorch":
            self.gr_processor = GraffitiProcessing(cfg)
        elif trt == "TensorRT": 
            if os.path.exists(cfg["graffiti_engine_path"]):
                from scripts.TRTGraffitiProcessor import TRTGraffitiProcessing
                self.gr_processor = TRTGraffitiProcessing(cfg)
            else:
                gr_processor = GraffitiProcessing(cfg)
                gr_processor.create_onnx()
                from build_trt.build_engine import EngineBuilder
                builder = EngineBuilder(False, 8)
                builder.create_network("weights/graffiti.onnx")
                if cfg["graffiti_engine_path"].split("_")[1].split(".")[0] == "fp16" or cfg["graffiti_engine_path"].split("_")[1].split(".")[0] == "fp32":
                    builder.create_engine(cfg["graffiti_engine_path"], cfg["graffiti_engine_path"].split("_")[1].split(".")[0])
                elif cfg["graffiti_engine_path"].split("_")[1].split(".")[0] == "int8":
                    builder.create_engine(cfg["graffiti_engine_path"], "int8", "data/images", "weights/calibration.cache", 123, 1)

                from scripts.TRTGraffitiProcessor import TRTGraffitiProcessing
                self.gr_processor = TRTGraffitiProcessing(cfg) #TODO
                
        self.dc_processor = DiscolorImagesProcessing(cfg)
        self.sd_processor = SurfaceDefects(cfg)

    def process_single_image(self, image_name):
        """Обработка одной картинки

        Args:
            image_name (str): Имя картинки из папки с картинками

        Returns:
            dict: Словарь с результатами
        """
        
        image_json_path = os.path.join(self.input_jsons, image_name[:-4] + ".json")
        if os.path.exists(image_json_path):

            with open(image_json_path, 'r') as file:
                image_json = json.load(file)

            image_path = os.path.join(self.images_path, image_name)
            image = cv2.imread(image_path)

            result_dc = self.dc_processor.image_process(image, image_json)
            result_sd = self.sd_processor.find_defects(image, image_json)
            result_gr = self.gr_processor.image_process(image, image_json)

            out = {"image_name": image_name,
                    "discolor_defect": result_dc,
                    "graffiti_defect": result_gr,
                    "surface_defect": result_sd}

            json_name = image_name[:-4] + ".json"
            with open(os.path.join(self.output, json_name), "w") as f:
                json.dump(out, f, indent=4)

            return out
        else:
            print("Did not find json : {}".format(image_json_path))
            return {"image_name": "",
                    "discolor_defect": "",
                    "graffiti_defect": "",
                    "surface_defect": ""}
