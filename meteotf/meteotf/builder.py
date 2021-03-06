import glob
import json
import os
import cv2
import sys
import random
import shutil
import builder_blueprint 

import feature
import tensorflow as tf
import tensorflow.train as tft

from editor import CropArea, TrainingImagePreview
from types import IntType, StringType, FloatType
# from PIL import Image

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
        self._image = cv2.cvtColor(self._image, cv2.COLOR_RGB2GRAY)
        self._image = cv2.resize(self._image, (0,0), fx=0.5, fy=0.5)

    def save(self, destination_path):
        assert type(destination_path) is StringType, 'destination_path: passed object of incorrect type'
        cv2.imwrite(destination_path, self._image)

class MeteoRecordWriter(object):
    def __init__(self, destination_path):
        assert type(destination_path) is StringType, 'destination_path: passed object of incorrect type'
        self._writer = tf.python_io.TFRecordWriter(destination_path)
        self._write_count = 0

    def write(self, example):
        assert type(example) is tft.Example, 'example: passed object of incorrect type'                        
        self._writer.write(example.SerializeToString())
        self._write_count += 1

        if not self._write_count % 1000:
            sys.stdout.flush()        

    def close(self):
        self._writer.close()        
        self._write_count = 0
        sys.stdout.flush()

class MeteoTrainingRecordWriter(object):
    def __init__(self, destination_dir, ratio):
        assert type(destination_dir) is StringType, 'destination_dir: passed object of incorrect type'
        assert type(ratio) is FloatType, 'ratio: passed object of incorrect type'        
        self._ratio = ratio        
        self._training_writer = MeteoRecordWriter(os.path.join(destination_dir, 'training.TFRecord'))
        self._validation_writer = MeteoRecordWriter(os.path.join(destination_dir, 'validation.TFRecord'))        

    def write(self, example):
        assert type(example) is tft.Example, 'example: passed object of incorrect type'
        writer = self._training_writer if random.random() <= self._ratio else self._validation_writer
        writer.write(example)

    def close(self):
        self._training_writer.close()
        self._validation_writer.close()
        sys.stdout.flush()

class MeteoTrainingSetBuilder(object):
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

    def build_tfrecord(self, training_dir, record_writer):        
        for class_dir in glob.glob(os.path.join(training_dir, "*")):
            for example_file in glob.glob(os.path.join(class_dir, "*")):                    
                example = feature.create_example(example_file)
                record_writer.write(example)

        record_writer.close()

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
            if item['accept_fn'](features):
                output_filename = training_image + '.jpeg'
                if not os.path.exists(source_path):
                    print('[ERROR] Path not exists:' + source_path)
                    return
                image = CroppedImage(source_path, item['crop_area'])                
                image.save(os.path.join(item['destination_dir'], output_filename))                    

    def _compile_blueprint(self, blueprint):
        """ Method creates directories required by blueprint items """
        for item in blueprint:
            if not os.path.exists(item['destination_dir']):
                os.makedirs(item['destination_dir'])
        
# Helper functions
def get_paths_for(operation):
    intermediate_path = '../data/{operation}-set'.format(operation=operation)
    source_images_path = '../data/{operation}-images/'.format(operation=operation)
    index_path = '../data/{operation}-set-index.json'.format(operation=operation)

    return intermediate_path, source_images_path, index_path

def rmifexists(path):
    if os.path.exists(path):
        shutil.rmtree(path)    

# Execution section 
if __name__ == "__main__":
    
    # HELP
    # python2.7 builder blueprint input_path index_path output_path intermediate_path
    # python2.7 builder.py wind ../data/training-images ../data/training-set-index.json ../data/wind-model/records/ ../data/tmp/intermediate-set

    # Preparing the training set
    blueprint_name = sys.argv[1]
    input_path = sys.argv[2]
    index_path = sys.argv[3]
    output_path = sys.argv[4] # '../data/wind-model/records/' # INPUT PARAMETER
    intermediate_path = sys.argv[5]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # intermediate_path, source_images_path, index_path = get_paths_for('training')
    blueprint = builder_blueprint.index[blueprint_name](intermediate_path)
    rmifexists(intermediate_path)

    builder = MeteoTrainingSetBuilder(input_path, index_path)    
    builder.build_intermediate_set(blueprint)
    builder.build_tfrecord(intermediate_path, MeteoTrainingRecordWriter(output_path, 0.8))    

    # Preparing the prediction sets
    # intermediate_path, source_images_path, index_path = get_paths_for('prediction')
    # blueprint = builder_blueprint.index[blueprint_name](intermediate_path)

    # for item in blueprint:        
    #    builder = MeteoTrainingSetBuilder(source_images_path, index_path)
    #    rmifexists(intermediate_path)
    #    builder.build_intermediate_set([item])        
    #    builder.build_tfrecord(intermediate_path, MeteoRecordWriter(
    #        os.path.join(output_path, item['class_name'] + '.TFRecord')
    #    ))