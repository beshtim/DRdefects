import numpy as np
import torchvision.models as models
import torch
from torchvision import transforms as T
from PIL import Image
import torch.nn as nn
import cv2.cv2 as cv2


class GraffitiProcessing:
    """Граффити процессор
    """
    def __init__(self, cfg: dict):
        self.weights_path = cfg["graffiti_classifier_weights"]
        self.device = torch.device('cuda')
        self.net = self.load_weights()
        self.transform = T.Compose([
            T.Resize((100, 100)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.from_int_to_type = {0: 'clean', 1: 'dirty'}
        self.from_type_to_int = {v: k for k, v in self.from_int_to_type.items()}        

    def load_weights(self):
        """Загрузка весов
        """
        net = models.resnet50()
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)
        net.load_state_dict(torch.load(self.weights_path, map_location="cpu"))
        net = net.to(self.device)
        net.eval()
        return net

    def image_process(self, image: np.ndarray, image_json):
        """Обработка картинки

        Args:
            image (np.ndarray): Фотография
            image_json (dict): json фотографии

        Returns:
            str: True/Falce - есть дефект или нет
        """
        bbox = image_json["bbox"][0]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.crop((x1, y1, x2, y2))
        img = self.transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            input = img.to(self.device)
            outputs = self.net(input)
            _, predictions = torch.max(outputs, 1)
            result = int(predictions.tolist()[0])

            if result == 1:
                return str(True)

            elif result == 0:
                return str(False)
