from scripts.MainProcessor import MainProcessor
from argparse import ArgumentParser
import os
import json
import pandas as pd
import multiprocessing as mp
from functools import partial

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-ii", "--input_imgs", help="path to imgs folder", default="data/images")
    parser.add_argument("-ij", "--input_jsons", help="path to jsons folder", default="data/jsons")
    parser.add_argument("-o", "--output", help="path to output folder", default="data/output")
    parser.add_argument("-mt", "--model_type", help="use (TensorRT/PyTorch)", default="PyTorch")

    return parser.parse_args()

def process_images(input_imgs, input_jsons, output, trt):
    print('Initializing...')

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(cfg_path, "r") as cfg_file:
        cfg = json.load(cfg_file)

    main_proc = MainProcessor(input_imgs, input_jsons, trt, output, cfg)
    
    os.makedirs(output, exist_ok=True)
    data_list = os.listdir(input_imgs)
    print("Processing...")

    df_results = pd.DataFrame()

    for i, image_name in enumerate(data_list):
        out = main_proc.process_single_image(image_name)
        df_results = df_results.append(out, ignore_index=True)
        print('\r[{}/{}]'.format(i+1, len(data_list)), end='')

    df_results.to_csv(os.path.join(output, "all_results.csv"))

    print(' Done!')

def main():
    args = parse_args()
    input_imgs = args.input_imgs
    input_jsons = args.input_jsons
    output = args.output
    trt = args.model_type
    if os.getenv('USE_TRT') == '1':
        trt = "TensorRT"
    else:
        trt = "PyTorch"
    print("Using model type:", trt)

    process_images(input_imgs=input_imgs, input_jsons=input_jsons, output=output, trt=trt)

if __name__ == "__main__":
    main()
