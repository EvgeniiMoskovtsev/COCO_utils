import os
import glob
import json
from PIL import Image, ImageDraw

#PATHS
path_train_images = os.path.abspath("D:\\dataset\\train_images")
path_train_jsons = os.path.abspath("D:\\dataset\\train_json")
path_train_masks = os.path.abspath("D:\\dataset\\train_masks")

#READ FILES
image_files = glob.glob(os.path.join(path_train_images+"\\leaf", "*.jpg"))
json_files = glob.glob(os.path.join(path_train_jsons, "*.json"))
mask_files = glob.glob(os.path.join(path_train_masks+"\\leaf", "*.jpg"))

#CREATE MASK FROM JSON
def filename_detection(path):
    filename_with_ext = os.path.basename(path)
    filename, file_extension = os.path.splitext(filename_with_ext)
    return filename

if len(mask_files)!=len(json_files):
    for i in range(0, len(json_files)):
        json_file = json_files[i]
        filename = filename_detection(json_file)
        with open(json_file, "r") as read_file:
            f = read_file.read()
            obj = json.loads(f)
        h = obj['imageHeight']
        w = obj['imageWidth']
        img = Image.new('RGB', (w, h))
        for j in range(0, len(obj['shapes'])):
            poly = obj['shapes'][j]['points']
            a = []
            for m in range(0, len(poly)):
                poly_temp = tuple(poly[m])
                a.append(poly_temp)
            ImageDraw.Draw(img).polygon(a, outline=(255, 255, 255), fill=(255, 255, 255))
        mask = np.array(img)
        x = array_to_img(mask)
        x.show()
        x.save(path_train_masks + "\\" + filename + ".jpg")