import os
import glob
import json
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw

#PATHS
path_train_images = os.path.abspath("D:\\dataset\\train_images")
path_train_jsons = os.path.abspath("D:\\dataset\\train_json")
path_train_masks = os.path.abspath("D:\\dataset\\train_masks")

#READ FILES
image_files = glob.glob(os.path.join(path_train_images+"\\leaf", "*.jpg"))
json_files = glob.glob(os.path.join(path_train_jsons, "*.json"))
assert len(image_files)==len(json_files)

#PREPARE TRAIN, TEST, VALIDATION SETS
#set size
train = 0.7
test = round((1-train)/2, 2)
validation = round(1-train-test, 2)

#divide images on sets
random.shuffle(image_files)
train_image_files = image_files[:round(len(image_files)*train)]
diff = np.setdiff1d(image_files, train_image_files)
test_image_files = diff[:round(len(diff)/2)]
val_image_files = diff[round(len(diff)/2):]
print(len(train_image_files), len(test_image_files), len(val_image_files))

#prepare json files for sets
def filename_detection(path):
    filename_with_ext = os.path.basename(path)
    filename, file_extension = os.path.splitext(filename_with_ext)
    return filename

def pick_json_files(files):
    set_json_files = []
    for i in range(0, len(files)):
        filename = filename_detection(files[i])
        set_json_files.append(path_train_jsons + '\\' + filename +'.json')
    return set_json_files

train_json_files = pick_json_files(train_image_files)
test_json_files = pick_json_files(test_image_files)
val_json_files = pick_json_files(val_image_files)
print(len(train_json_files), len(test_json_files), len(val_json_files))