import json


class ParseCoco(object):
    def __init__(self, path_to_package_images=None, path_to_json=None, type_of_labels=None):
        if type_of_labels == 'COCO':
            self.path_to_images_package = path_to_package_images
            self.path_to_json = path_to_json
        else:
            raise Exception("Incorrect type of labels")

    def load_data(self):
        with open(self.path_to_json, 'r') as file:
            data = file.read()
        obj = json.loads(data)
        image_annotations = obj['images']
        return obj, image_annotations

    def parse_image_filename(self, image_id, image_annotations):
        for obj in image_annotations:
            if obj['id'] == image_id:
                return obj['file_name']

    def adapt_coorinates_to_imgaug(self, polygon):
        tuple_x_y = []
        for i in range(0, len(polygon), 2):
            x, y = (polygon[i:i + 2])
            tuple_x_y.append((int(x), int(y)))
        return tuple_x_y

    def main(self):
        obj, image_annotations = self.load_data()
        data = dict()
        for annotation in obj['annotations']:
            coordinates = annotation['segmentation']
            image_id = annotation['image_id']
            filename = self.parse_image_filename(image_id, image_annotations)
            tuple_x_y = self.adapt_coorinates_to_imgaug(coordinates[0])
            data.setdefault(filename, [])
            data[filename].append(tuple_x_y)
        return data

    # def __iter__(self):

    def __call__(self):
        return self.main()


if __name__ == "__main__":
    parse_class = ParseCoco(path_to_package_images=r'C:\Users\apelv\Desktop\utils\labelme2coco\clear_leafs_images',
                            path_to_json=r"C:\Users\apelv\Desktop\utils\labelme2coco\trainval.json",
                            type_of_labels='COCO')()

    print(parse_class)
