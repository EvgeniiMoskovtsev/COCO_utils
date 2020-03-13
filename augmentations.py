from leafs.parsecoco import ParseCoco
from imgaug.augmentables.polys import Polygon
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import os
from tqdm import tqdm
import yaml
from leafs.writecoco import WriteCoco

class AugPolygons:
    def load_config(self, path_to_config):
        with open(path_to_config, 'r') as ymlfile:
            cfg = yaml.full_load(ymlfile)
            return cfg

    def __init__(self, path_to_config):
        self.cfg = self.load_config(path_to_config)
        self.PATH_TO_PACKAGE_IMAGES = cfg['images']['path']
        self.PATH_TO_JSON = cfg['annotations']['path']
        self.TYPE_OF_LABELS = cfg['annotations']['type']

    def augment(self):
        data = ParseCoco(path_to_package_images=PATH_TO_PACKAGE_IMAGES,
                         path_to_json=PATH_TO_JSON,
                         type_of_labels=TYPE_OF_LABELS)()
        loop = 0
        while loop != 5:
            for filename, polygons in tqdm(data.items()):
                image = imageio.imread(os.path.join(PATH_TO_PACKAGE_IMAGES, filename))
                list_of_poly_objects = []
                for polygon in polygons:
                    poly_object = Polygon(polygon)
                    list_of_poly_objects.append(poly_object)

                psoi = ia.PolygonsOnImage(list_of_poly_objects, shape=image.shape)
                aug = iaa.Sequential([
                    iaa.AdditiveGaussianNoise(scale=(0, 50)),
                    iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                    # iaa.AddToHueAndSaturation((-50, 50)),
                    iaa.Add((-40, 40)),
                    iaa.Affine(rotate=(-20, 20), translate_percent={"x": 0.1, "y": 0.1}, scale=(0.5, 1.8)),
                    iaa.Fliplr(1.0)
                ])
                image_aug, psoi_aug = aug(image=image, polygons=psoi)
                # new_name = filename.split('.')[0]
                cv2.imwrite("augmented/{}loop{}.jpg".format(filename, LOOPS), image_aug)
                WriteCoco()
            loop += 1


if __name__ == "__main__":
    with open("configs/config.yml", 'r') as ymlfile:
        cfg = yaml.full_load(ymlfile)

    PATH_TO_PACKAGE_IMAGES = cfg['images']['path']
    PATH_TO_JSON = cfg['annotations']['path']
    TYPE_OF_LABELS = cfg['annotations']['type']
    LOOPS = 0

    data = ParseCoco(path_to_package_images=PATH_TO_PACKAGE_IMAGES,
                     path_to_json=PATH_TO_JSON,
                     type_of_labels=TYPE_OF_LABELS)()

    while LOOPS!=3:
        for filename, polygons in tqdm(data.items()):
            image = imageio.imread(os.path.join(PATH_TO_PACKAGE_IMAGES, filename))
            list_of_poly_objects = []
            for polygon in polygons:
                poly_object = Polygon(polygon)
                list_of_poly_objects.append(poly_object)

            psoi = ia.PolygonsOnImage(list_of_poly_objects, shape=image.shape)
            aug = iaa.Sequential([
                iaa.AdditiveGaussianNoise(scale=(0, 50)),
                iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                # iaa.AddToHueAndSaturation((-50, 50)),
                iaa.Add((-40, 40)),
                iaa.Affine(rotate=(-20, 20), translate_percent={"x": 0.1, "y": 0.1}, scale=(0.5, 1.8)),
                iaa.Fliplr(1.0)
            ])
            image_aug, psoi_aug = aug(image=image, polygons=psoi)
            # new_name = filename.split('.')[0]
            cv2.imwrite("augmented/{}loop{}.jpg".format(filename, LOOPS), image_aug)
            WriteCoco(image_aug.shape, psoi_aug, filename)()
        LOOPS+=1
            # cv2.waitKey(0)
            # ia.imshow(psoi_aug.draw_on_image(image_aug, alpha_face=0.2, size_points=7, color_face=(255,0,0)))
            # for polygon in polygons:
