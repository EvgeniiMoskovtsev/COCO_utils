import os
import glob
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw


#PATHS
path_train_images = os.path.abspath("D:\\dataset\\train_images")
path_train_jsons = os.path.abspath("D:\\dataset\\train_json")
path_train_masks = os.path.abspath("D:\\dataset\\train_masks")

#READ FILES
image_files = glob.glob(os.path.join(path_train_images+"\\leaf", "*.jpg"))
json_files = glob.glob(os.path.join(path_train_jsons, "*.json"))
mask_files = glob.glob(os.path.join(path_train_masks+"\\leaf", "*.jpg"))

def get_min_image_size (image_files):
    h = []
    w = []
    for i in range (0, len(image_files)):
        image_file = image_files[i]
        image = cv2.imread(image_file)
        h_temp, w_temp, ch_temp = image.shape
        h.append(h_temp)
        w.append(w_temp)
    h_min = np.min(h)
    w_min = np.min(w)
    return h_min, w_min

def create_img_mask_arrays(json_files, image_files):
    assert len(json_files)==len(image_files)
    img = None
    mask = None
    h_resize, w_resize, = get_min_image_size(image_files)
    for i in range(0, len(json_files)):
        json_file = json_files[i]
        img_file = image_files[i]
        img_temp = cv2.imread(img_file)
        img_temp = cv2.resize(img_temp, dsize=(w_resize, h_resize), interpolation=cv2.INTER_CUBIC)
        img_temp = np.array([img_temp])
        if img is None:
            img = img_temp
        else:
            img = np.concatenate((img, img_temp), axis=0)
        with open(json_file, "r") as read_file:
            f = read_file.read()
            obj = json.loads(f)
        h = obj['imageHeight']
        w = obj['imageWidth']
        download_img = Image.new('RGB', (w, h))
        for j in range(0, len(obj['shapes'])):
            poly = obj['shapes'][j]['points']
            a = []
            for m in range(0, len(poly)):
                poly_temp = tuple(poly[m])
                a.append(poly_temp)
            ImageDraw.Draw(download_img).polygon(a, outline=(255, 255, 255), fill=(255, 255, 255))
        mask_temp = np.array(download_img)
        mask_temp = cv2.resize(mask_temp, dsize=(w_resize, h_resize), interpolation=cv2.INTER_CUBIC)
        mask_temp = np.array([mask_temp])
        if mask is None:
            mask = mask_temp
        else:
            mask = np.concatenate((mask, mask_temp), axis=0)
    return img, mask

#CREATE IMAGE AND MASK ARRAYS
img_array, mask_array = create_img_mask_arrays(json_files, image_files)