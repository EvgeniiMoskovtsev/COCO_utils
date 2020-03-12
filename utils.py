import os
import json

class IsFileExist:
    def __call__(self, filename):
        if not os.path.exists(filename):
            raise Exception("Not founded json")
        else:
            pass

class CreateCarcass:
    def __call__(self, filename):
        data = json.load(filename)
        print(data)


class DumpToJson:
    def __init__(self, data_json):
        self.last_id = 0
        self.data = json.loads(data)

    def find_next_id(self):
        if self.last_id == 0:
            for image in self.data['images']:
                if image['id'] is None:
                    pass
                else:
                    if self.last_id < image['id']:
                        self.last_id = image['id']
                        next_id = self.last_id + 1
        return next_id or self.last_id

    def dump(self, filename, image_shape):
        pass


    def __call__(self):
        pass

if __name__ == "__main__":
    with open(r"C:\Users\apelv\Desktop\utils\labelme2coco\trainval.json", 'r') as file:
        data = file.read()
    DumpToJson = DumpToJson(data)()