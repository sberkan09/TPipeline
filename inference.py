import os
import sys
import argparse
from utils import *
from detr.detr import Detr

parser = argparse.ArgumentParser()

# Command Line Interface Arguments
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--device', type=str, help='Example: --device cuda:0', required=False, default='cpu')

args = parser.parse_args()

data_path = args.data_path

inference_data_files = [file for file in os.listdir(data_path) if file.split('.')[-1] == 'dcm']

if len(inference_data_files) <= 0:
    sys.exit(1)

detr = Detr()

for file in inference_data_files:
    img = open_img(os.path.join(data_path, file))
    img = preprocess(img)
    bbox_list = detr.run(img)
    print(bbox_list)