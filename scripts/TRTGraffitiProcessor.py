import os
import sys
import time
import ctypes
import argparse
import numpy as np
import tensorrt as trt
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
from torchvision import transforms as T

class TRTGraffitiProcessing:

    def __init__(self, cfg: dict):
        """
        Args:
            cfg (dict): Конфиг файл, в нем должен быть путь к сериализованному движку для загрузки с диска (graffiti_engine_path).
        """
        self.engine_path = cfg["graffiti_engine_path"]
        self.transform = T.Compose([
            T.Resize((100, 100)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.from_int_to_type = {0: 'clean', 1: 'dirty'}
        self.from_type_to_int = {v: k for k, v in self.from_int_to_type.items()}


        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """Получение спецификаций для входного тензора сети.

        Returns:
            _type_: Два элемента, форма входного тензора и его (numpy) тип данных.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self) -> dict:
        """Получение спецификаций для выходных тензоров сети.

        Returns:
            dict: Список с двумя элементами на каждый выход: форма и (numpy) тип данных.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def image_process(self, image: np.ndarray, image_json):
        """Обработка картинки

        Args:
            image (np.ndarray): Фотография
            image_json (dict): json фотографии

        Returns:
            str: True/False - есть дефект или нет
        """
        bbox = image_json["bbox"][0]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.crop((x1, y1, x2, y2))
        img = self.transform(img)
        img = img.unsqueeze(0)
        img = img.numpy()

        # Prepare the output data
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(img))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])

        # Process the results
        type_gr = outputs[0]
        
        result = np.argmax(type_gr)
        if result == 1:
            return str(True)

        elif result == 0:
            return str(False)