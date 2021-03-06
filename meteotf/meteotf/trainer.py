import os
import sys
import feature
import tfcoreml
import tensorflow as tf
import tensorflow.train as tft
import tensorflow.compat as tfc

class MeteoMLModel(object):

    def __init__(self, output_path):
        self._model = tf.estimator.DNNClassifier(
            hidden_units=[],
            n_classes=3,
            feature_columns=feature.feature_columns,
            model_dir=output_path
        )
    
    def train(self, training_set, epochs=20, steps=8000):        
        training_dataset = self._prepare_dataset(training_set, epochs)
        self._model.train(lambda:self._input_function(training_dataset), steps=steps)

    def evaluate(self, validation_set):
        validation_dataset = self._prepare_dataset(validation_set, 1)
        test_accuracy = self._model.evaluate(lambda:self._input_function(validation_dataset))['accuracy']

        print('Test accuracy:', test_accuracy)
        return test_accuracy

    def predict(self, prediction_set, expected_class):
        prediction_dataset = self._prepare_dataset(prediction_set, 1)
        predictions = self._model.predict(lambda:self._input_function(prediction_dataset))
        self._print_prediction_summary(predictions, expected_class)       

    def save(self, export_dir):       
        input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn({
            'image/encoded': tf.FixedLenFeature([90 * 42], tf.string)
        })

        exported_path =  self._model.export_savedmodel(export_dir, input_fn, as_text=False)        
        self._save_frozen_graph(exported_path, os.path.join(exported_path, 'frozen_model.pb'))

    def _save_frozen_graph(self, export_dir, output_path):
        with tf.Session(graph=tf.Graph()) as session:
            tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], export_dir)
            output_nodes = ['dnn/head/predictions/probabilities']            

            graph_def = session.graph.as_graph_def()                        
            output_graph_def = tf.graph_util.convert_variables_to_constants(session, graph_def, output_nodes)

            with tf.gfile.GFile(output_path, "wb") as file:
                file.write(output_graph_def.SerializeToString())            
                file.close()
                sys.stdout.flush() 
                print('Exported: Frozen graph')                    

    def _prepare_dataset(self, record_files, num_epochs):
        dataset = tf.data.TFRecordDataset(record_files)
        dataset = dataset.map(feature.parse_record)
        dataset = dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True)
        dataset = dataset.batch(30)
        dataset = dataset.repeat(num_epochs)
        
        print('Initialized: Dataset')
        return dataset

    def _input_function(self, dataset):
        return dataset.make_one_shot_iterator().get_next()

    def _print_prediction_summary(self, predictions, class_id):
        total_predictions = 0
        correct_predictions = 0        

        for prediction in predictions:
            if total_predictions == 0:
                print(prediction)
            total_predictions += 1
            if int(prediction['classes'][0]) == class_id:
                correct_predictions += 1

        print('Prediction summary for class: {class_id}: {correct} out of {total} total predictions'.format(
            correct=correct_predictions,
            total=total_predictions,
            class_id=class_id,
        ))

# Execution section
if __name__ == "__main__":     
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # HELP
    # python2.7 trainer.py input_path output_path
    # python2.7 trainer.py ../data/wind-model/records/ ../data/wind-model/saved-models

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    training_records = [os.path.join(input_path, 'training.TFRecord')]
    evaluation_records = [os.path.join(input_path, 'validation.TFRecord')]
            
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    meteo_model = MeteoMLModel(output_path)   
    meteo_model.train(training_records)            
    test_accuracy =  meteo_model.evaluate(evaluation_records)

    if test_accuracy < 0.8:
        print('Model NOT saved, test accuracy too low')
        exit(0)

    meteo_model.save(output_path)

    # for item in sys.argv[3:]:
    #    class_name = item.split(':')[0]
    #    label = int(item.split(':')[1])        
    #    meteo_model.predict([os.path.join(input_path, class_name + '.TFRecord')], label)