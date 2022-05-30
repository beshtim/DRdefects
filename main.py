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
    parser.add_argument("-icsv", "--input_csv", help="path to csv file", default="data/images_jsons.csv")
    parser.add_argument("-o", "--output", help="path to output folder", default="data/output")
    parser.add_argument("-mt", "--model_type", help="use (TensorRT/PyTorch)", default="PyTorch")

    return parser.parse_args()

def process_images(input_imgs, input_jsons, input_csv, output, trt):
    print('Initializing...')
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(cfg_path, "r") as cfg_file:
        cfg = json.load(cfg_file)

    df_input = pd.read_csv(input_csv, sep=";", index_col=0)

    main_proc = MainProcessor(input_imgs, input_jsons, input_csv, df_input, trt, output, cfg)
    
    os.makedirs(output, exist_ok=True)
    data_list = os.listdir(input_imgs)
    print("Processing...")
 
    for i, image_name in enumerate(data_list):
        main_proc.process_single_image(image_name)
        print('\r[{}/{}]'.format(i+1, len(data_list)), end='')

    print(' Done!')

def main():
    args = parse_args()
    input_imgs = args.input_imgs
    input_jsons = args.input_jsons
    input_csv = args.input_csv
    output = args.output
    trt = args.model_type
    print("Using model type:", trt)

    process_images(input_imgs=input_imgs, input_jsons=input_jsons, input_csv=input_csv, output=output, trt=trt)

if __name__ == "__main__":
    main()
