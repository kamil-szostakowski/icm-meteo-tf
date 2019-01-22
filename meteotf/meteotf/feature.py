import os
import types
import tensorflow as tf
import tensorflow.train as tft
import tensorflow.compat as tfc

def _int64_feature(value):  
  if not isinstance(value, list):
    value = [value]
  return tft.Feature(int64_list=tft.Int64List(value=value))

def _bytes_feature(value):  
  return tft.Feature(bytes_list=tft.BytesList(value=[value]))

def _get_tf_class(class_dir):
    class_label =  int(os.path.basename(class_dir).split('_')[-1])
    class_name = str(os.path.basename(class_dir).split('_')[0])    
    return class_label, class_name

feature_spec = {
  'image/label': tf.FixedLenFeature([], tf.int64),
  'image/encoded': tf.FixedLenFeature([], tf.string),
}

feature_columns = [
  tf.feature_column.numeric_column('image/encoded', shape=[90 * 42])
]

def create_example(image_path):
    assert type(image_path) is types.StringType, 'image_path: passed object of incorrect type'
        
    image_data = open(image_path, 'rb').read()
    class_label, class_name = _get_tf_class(os.path.split(image_path)[0])

    return tft.Example(features=tft.Features(feature={                 
        'image/label': _int64_feature(class_label),        
        'image/encoded': _bytes_feature(tfc.as_bytes(image_data)),
    }))

def parse_record(record):        
    parsed = tf.parse_single_example(record, feature_spec)
    image = tf.image.decode_jpeg(parsed['image/encoded'])
    image = tf.reshape(image, [90 * 42])
    print(image)
    label = tf.cast(parsed['image/label'], tf.int64)
    
    return { 'image/encoded': image }, label

# Something is wrong here
def serving_input_receiver_fn():  
  serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors') #, shape=[30*3780], name='input_example_tensor'
  receiver_tensors = {'inputs': serialized_tf_example}  

  features = tf.parse_example(serialized_tf_example, {
    'x': tf.FixedLenFeature([], tf.string)
  })

  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)