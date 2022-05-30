#!/bin/bash

arg_tag=defectdetector
MODEL_TYPE=PyTorch
arg_help=0
OUTPUT="$(pwd)/output"

# DEFAULT
IMAGES="/home/nanosemantics/beshkurov/boost_ex/defectdetector/data/images"
JSONS="/home/nanosemantics/beshkurov/boost_ex/defectdetector/data/jsons"
WEIGHTS="/home/nanosemantics/beshkurov/DEFDATA/weights" 

# ARGS 
while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--images)
      IMAGES="$2"
      shift;;
    -j|--jsons)
      JSONS="$2"
      shift;;
    -w|--weights)
      WEIGHTS="$2"
      shift;;
    -o|--output)
      MODEL_TYPE="$2"
      shift;;
    -mt|--model-type)
      MODEL_TYPE="$2"
      shift;;
    -h|--help) arg_help=1;;
  *) echo "Unknown parameter passed: $1"; echo "For help type: $0 --help"; exit 1;
esac; shift; done

if [ "$arg_help" -eq "1" ]; then
    echo "Usage: $0 [options]"
    echo " --help    or -h         : Print this help menu."
    echo " --images  or -i         : Path to images for docker volume. "
    echo " --jsons   or -j         : Path to jsons for docker volume. "
    echo " --weights or -w         : Path to weights folder with .pth file for torch and .trt for TensorRT inference for docker volume."
    echo " --output  or -o         : Path to output folder. DEFAULT: ./output"
    echo " --model-type or -mt     : <PyTorch/TensorRT> for graffiti processor.  DEFAULT: PyTorch"
    exit;
fi

arg_tag=defectdetector
name=defectdetector

docker_args="--name $name --rm -it -v $IMAGES:/defectdetector/data/images -v $JSONS:/defectdetector/data/jsons -v $WEIGHTS:/defectdetector/weights -v $OUTPUT:/defectdetector/data/output $arg_tag:latest -mt $MODEL_TYPE"

echo "--------------------------------"
echo "Launching container:"
echo "> docker run --runtime=nvidia  $docker_args"
docker run --runtime=nvidia $docker_args
