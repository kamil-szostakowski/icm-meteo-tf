import glob
import json
import os
import cv2

from editor import CropArea, TrainingImagePreview
from types import IntType, StringType

class CroppedImage(object):
    def __init__(self, img_path, crop):
        """
        Args:
            img_path (str): full path to the full meteorogram image.
            crop (CropArea): area which should be cropped out of the meteorogram in order to prepare a training example.
        """
        
        assert type(img_path) is StringType, 'img_path: passed object of incorrect type'
        assert type(crop) is CropArea, 'crop: passed object of incorrect type'
        
        self._image = cv2.imread(img_path)        
        self._image = self._image[crop.y_slice, crop.x_slice]

    def save(self, destination_path):
        assert type(destination_path) is StringType, 'destination_path: passed object of incorrect type'
        cv2.imwrite(destination_path, self._image)        

class TrainingSetBuilder(object):
    def __init__(self, images_path, index_path):
        assert type(images_path) is StringType, 'images_path: passed object of incorrect type'
        assert type(index_path) is StringType, 'index_path: passed object of incorrect type'

        self._images_path = images_path
        self._index_path = index_path
        self._index = { 'sorted_keys': [], 'values': {} , 'active_index': 0 }

        self._load_index(index_path)

    def build(self, blueprint):
        index = 1
        self._compile_blueprint(blueprint)

        for image_key in self._index['sorted_keys']:
            if image_key in self._index['values']:
                features = self._index['values'][image_key].encode('ascii','ignore')
                training_image = image_key.encode('ascii','ignore')

                print('%d Processing item %s -> %s' % (index, training_image, features))
                self._build_item(training_image, features, blueprint)                
                index += 1

    def _load_index(self, index_path):
        """ Method loads features index from a file  """

        if not os.path.exists(index_path):
            raise ValueError('File or directory %s does not exists' % (index_path))

        with open(index_path) as infile:
            self._index = json.load(infile)

    def _build_item(self, training_image, features, blueprint):
        """ Method builds training examples based on meteorogram and blueprint """
        source_path = os.path.join(self._images_path, training_image + '.png')
        
        for item in blueprint:
            if not item['feature'].encode('ascii','ignore') in features:
                continue

            output_filename = training_image + '@' + features + '.png'
            image = CroppedImage(source_path, item['crop_area'])
            image.save(os.path.join(item['destination_dir'], output_filename))            

    def _compile_blueprint(self, blueprint):
        """ Method creates directories required by blueprint items """
        for item in blueprint:
            if not os.path.exists(item['destination_dir']):
                os.makedirs(item['destination_dir'])
        
# Execution section
if __name__ == "__main__":
    builder = TrainingSetBuilder('../source-images/', '../training-set-index.json')
    builder.build([
        {
            'crop_area': CropArea(65, 140, 180, 85),
            'destination_dir': '../training-set/rain/',
            'feature': 'R'
        },
        {
            'crop_area': CropArea(65, 140, 180, 85),
            'destination_dir': '../training-set/snow/',
            'feature': 'S'
        },
        {
            'crop_area': CropArea(65, 140, 180, 85),
            'destination_dir': '../training-set/storm/',
            'feature': 'T'
        },                
        {
            'crop_area': CropArea(65, 314, 180, 85),
            'destination_dir': '../training-set/wind/',
            'feature': 'W'
        },
        {
            'crop_area': CropArea(65, 522, 180, 85),
            'destination_dir': '../training-set/clouds/',
            'feature': 'C'
        }
    ])