import glob
import json
import os
import cv2
import sys
import random

import tensorflow as tf
import tensorflow.train as tft
import tensorflow.compat as tfc

from editor import CropArea, TrainingImagePreview
from types import IntType, StringType, FloatType

def _int64_feature(value):  
  if not isinstance(value, list):
    value = [value]
  return tft.Feature(int64_list=tft.Int64List(value=value))

def _bytes_feature(value):  
  return tft.Feature(bytes_list=tft.BytesList(value=[value]))

feature_description = {
    'image/class/label': tf.FixedLenFeature([], tf.int64),
    'image/class/text': tf.VarLenFeature(tf.string),
    'image/filename': tf.FixedLenFeature([], tf.string),
    'image/format': tf.FixedLenFeature([], tf.string),
    'image/encoded': tf.FixedLenFeature([], tf.string),
}  

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
        self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        self._image = cv2.resize(self._image, (0,0), fx=0.5, fy=0.5)

    def save(self, destination_path):
        assert type(destination_path) is StringType, 'destination_path: passed object of incorrect type'
        cv2.imwrite(destination_path, self._image)

class TFWriter(object):
    def __init__(self, destination_dir, ratio):
        assert type(destination_dir) is StringType, 'destination_dir: passed object of incorrect type'
        assert type(ratio) is FloatType, 'ratio: passed object of incorrect type'
        
        self._ratio = ratio
        self._write_count = 0        
        self._training_writer = tf.python_io.TFRecordWriter(destination_dir + 'training.TFRecord')
        self._validation_writer = tf.python_io.TFRecordWriter(destination_dir + 'validation.TFRecord')        

    def write(self, example):
        assert type(example) is tft.Example, 'example: passed object of incorrect type'        
        
        writer = self._training_writer if random.random() <= self._ratio else self._validation_writer
        writer.write(example.SerializeToString())
        self._write_count += 1

        if not self._write_count % 1000:
            sys.stdout.flush()

    def close(self):
        self._training_writer.close()
        self._validation_writer.close()
        self._write_count = 0
        sys.stdout.flush()

class TrainingSetBuilder(object):
    def __init__(self, images_path, index_path):
        assert type(images_path) is StringType, 'images_path: passed object of incorrect type'
        assert type(index_path) is StringType, 'index_path: passed object of incorrect type'

        self._images_path = images_path
        self._index_path = index_path
        self._index = { 'sorted_keys': [], 'values': {} , 'active_index': 0 }
        self._load_index(index_path)

    def build_intermediate_set(self, blueprint):
        index = 1
        self._compile_blueprint(blueprint)

        for image_key in self._index['sorted_keys']:
            if image_key in self._index['values']:
                features = self._index['values'][image_key].encode('ascii','ignore')
                training_image = image_key.encode('ascii','ignore')

                print('%d Processing item %s -> %s' % (index, training_image, features))
                self._build_item(training_image, features, blueprint)                
                index += 1

    def build_tfrecord(self, training_dir, destination_dir, ratio):
        self._record_writer = TFWriter(destination_dir, ratio)

        for class_dir in glob.glob(os.path.join(training_dir, "*")):
            class_label, class_name =  self._get_tf_class(class_dir)

            for example_file in glob.glob(os.path.join(class_dir, "*")):
                image_string = open(example_file, 'rb').read()

                example = tft.Example(features=tft.Features(feature={                 
                    'image/class/label': _int64_feature(class_label),
                    'image/class/text': _bytes_feature(tfc.as_bytes(class_name)),
                    'image/filename': _bytes_feature(tfc.as_bytes(os.path.basename(example_file))),
                    'image/format': _bytes_feature(tfc.as_bytes('JPEG')),
                    'image/encoded': _bytes_feature(image_string),
                }))
                self._record_writer.write(example)

        self._record_writer.close()                

    def _get_tf_class(self, class_dir):
        class_label =  int(os.path.basename(class_dir).split('_')[-1])
        class_name = str(os.path.basename(class_dir).split('_')[0])
        return class_label, class_name

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

            output_filename = training_image + '.jpeg'
            image = CroppedImage(source_path, item['crop_area'])
            image.save(os.path.join(item['destination_dir'], output_filename))            

    def _compile_blueprint(self, blueprint):
        """ Method creates directories required by blueprint items """
        for item in blueprint:
            if not os.path.exists(item['destination_dir']):
                os.makedirs(item['destination_dir'])
        
# Execution section
if __name__ == "__main__":
    builder = TrainingSetBuilder('../data/source-images/', '../data/training-set-index.json')
    builder.build_intermediate_set([
        {
            'crop_area': CropArea(65, 140, 180, 85),
            'destination_dir': '../data/training-set/rain_0/',
            'feature': 'R'
        },
        # {
        #     'crop_area': CropArea(65, 140, 180, 85),
        #     'destination_dir': '../data/training-set/snow/',
        #     'feature': 'S'
        # },
        # {
        #     'crop_area': CropArea(65, 140, 180, 85),
        #     'destination_dir': '../data/training-set/storm/',
        #     'feature': 'T'
        # },                
        {
            'crop_area': CropArea(65, 314, 180, 85),
            'destination_dir': '../data/training-set/wind_1/',
            'feature': 'W'
        },
        {
            'crop_area': CropArea(65, 522, 180, 85),
            'destination_dir': '../data/training-set/clouds_2/',
            'feature': 'C'
        }
    ])
    builder.build_tfrecord('../data/training-set/', '../data/', 0.8)            