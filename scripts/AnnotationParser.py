import json
import pandas as pd
import os


class JsonParser:
    """Обработка json файла
    """
    def __init__(self, jsons_path, csv_path):
        self.jsons_path = jsons_path
        self.df = pd.read_csv(csv_path, sep=";", index_col=0)

    def get_json_name(self, image_name):
        return self.df[self.df["name"] == image_name]["json"].values[0]

    def read_json(self, image_name):
        json_name = self.get_json_name(image_name)
        json_path = os.path.join(self.jsons_path, json_name)
        with open(json_path, 'r') as file:
            data = json.load(file)
        return data


class AnnParser:
    """Обработка json файла в разметке coco
    """
    def __init__(self, annotation_path: str):
        self.annotation_path = annotation_path

        with open(self.annotation_path, 'r') as file:
            self.full_coco_json = json.load(file)

    def get_image_bboxes(self, image_name):
        image_anns = self.get_image_annotations(image_name)
        all_bb = []
        for ann in image_anns:
            all_bb.append(ann["bbox"])
        return all_bb

    def get_image_id_from_name(self, image_name: str) -> int:
        for image_desc in self.full_coco_json["images"]:
            if image_desc["file_name"] == image_name:
                image_id = image_desc["id"]
                return image_id

    def get_image_annotations(self, image_name: str) -> list:
        image_id = self.get_image_id_from_name(image_name)
        image_anns = []
        for annotations in self.full_coco_json["annotations"]:
            if annotations["image_id"] == image_id:
                image_anns.append(annotations)
        return image_anns

    def get_all_image_segmentations(self, image_name: str) -> list:
        image_anns = self.get_image_annotations(image_name)
        all_segments = []
        for ann in image_anns:
            all_segments.append(ann["segmentation"][0])
        return all_segments
