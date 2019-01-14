import io
import feature
import tensorflow as tf

from PIL import Image

if __name__ == "__main__":
    raw_image_dataset = tf.data.TFRecordDataset('../data/training.TFRecord')

    def _parse_image_function(example_proto):
        return tf.parse_single_example(example_proto, feature.feature_description())

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    element = parsed_image_dataset.make_one_shot_iterator().get_next()
    
    with tf.Session() as sess:
        raw_image = sess.run(element)['image/encoded']
        image = Image.open(io.BytesIO(raw_image))
        image.show()