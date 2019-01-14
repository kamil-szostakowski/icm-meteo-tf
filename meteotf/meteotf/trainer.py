#import types
import feature
import tensorflow as tf
import tensorflow.train as tft
import tensorflow.compat as tfc

class MeteoMLModel(object):

    def __init__(self):
        # 200, 100, 90
        self._model = tf.estimator.DNNClassifier([200,100,90],n_classes=3,feature_columns=feature.feature_columns())
    
    def train(self, training_set):        
        self._training_dataset = self._prepare_dataset(training_set, 1)
        self._model.train(lambda:self._input_function(self._training_dataset), steps=1000)

    def evaluate(self, validation_set):
        self._validation_dataset = self._prepare_dataset(validation_set, 1)
        test_acc = self._model.evaluate(lambda:self._input_function(self._validation_dataset))['accuracy']
        print('Test accuracy:', test_acc)

    def _prepare_dataset(self, record_files, num_epochs):
        dataset = tf.data.TFRecordDataset(record_files)
        dataset = dataset.map(feature.parse_record)
        dataset = dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True)
        dataset = dataset.batch(30)
        dataset = dataset.repeat(num_epochs)
        
        print('Dataset initialized')
        return dataset

    def _input_function(self, dataset):
        return dataset.make_one_shot_iterator().get_next()

# Execution section
if __name__ == "__main__": 
    tf.logging.set_verbosity(tf.logging.INFO)   
    trainer = MeteoMLModel()
    trainer.train(['../data/training.TFRecord'])
    trainer.evaluate(['../data/validation.TFRecord'])