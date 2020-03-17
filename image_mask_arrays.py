import os
import glob
import json
import random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
from PIL import Image, ImageDraw
import imgaug as ia
from imgaug import augmenters as iaa
import imageio as io

#PATHS
path_train_images = os.path.abspath("D:\\dataset\\train_images")
path_train_jsons = os.path.abspath("D:\\dataset\\train_json")
path_train_masks = os.path.abspath("D:\\dataset\\train_masks")
#path_aug_images = os.path.abspath("D:\\dataset\\train_augmentation\\images\\")
#path_aug_masks = os.path.abspath("D:\\dataset\\train_augmentation\\masks\\")
#path_test = os.path.abspath("D:\\dataset\\test\\")

#READ FILES
image_files = glob.glob(os.path.join(path_train_images+"\\leaf", "*.jpg"))
json_files = glob.glob(os.path.join(path_train_jsons, "*.json"))
mask_files = glob.glob(os.path.join(path_train_masks+"\\leaf", "*.jpg"))

#CREATE MASK FROM JSON
def create_img_mask_arrays(json_files, image_files):
    img = None
    mask = []
    assert len(json_files)==len(image_files)
    for i in range(0, len(json_files)):
        json_file = json_files[i]
        img_file = image_files[i]
        img_temp = cv2.imread(img_file)
        #img_temp = load_img(img_file)
        #img_temp = img_temp.transpose(Image.ROTATE_270)
        print(type(img_temp))
        print(img_temp.shape)
        #img_temp = img_to_array(img_temp)
        #print(type(img_temp))
        #img_temp = np.resize(img_temp, (1080, 1920))
        img_temp = cv2.resize(img_temp, dsize=(1080, 1920), interpolation=cv2.INTER_CUBIC)
        print(img_temp.shape)
        img_temp = np.array([img_temp])
        print(img_temp.shape)
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
        mask.append(mask_temp)
    return img, mask

img_array, mask_array = create_img_mask_arrays(json_files, image_files)



img = Image.fromarray(img_array[0], 'RGB')
img.show()