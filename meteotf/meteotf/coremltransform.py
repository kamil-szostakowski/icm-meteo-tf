import tfcoreml
import tensorflow as tf

from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

if __name__ == "__main__":
    # Load the TF graph definition
    model_dir = '../data/saved-models/1548106781'
    tf_model_path = '{root_dir}/frozen_model.pb'.format(root_dir=model_dir)

    with open(tf_model_path, 'rb') as in_file:
        serialized = in_file.read()

    tf.reset_default_graph()
    original_gdef = tf.GraphDef()
    original_gdef.ParseFromString(serialized)
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(original_gdef, name='')
        ops = graph.get_operations()

    # Strip the JPEG decoder and preprocessing part of TF model
    input_node_names = ['dnn/input_from_feature_columns/input_layer/image/encoded/ToFloat']
    output_node_names = ['dnn/head/predictions/probabilities']

    gdef = strip_unused_lib.strip_unused(
        input_graph_def = original_gdef,
        input_node_names = input_node_names,
        output_node_names = output_node_names,
        placeholder_type_enum = dtypes.float32.as_datatype_enum
    )

    # Save it to an output file
    frozen_model_file = '{root_dir}/inception_v3.pb'.format(root_dir=model_dir)
    with gfile.GFile(frozen_model_file, "wb") as out_file:
        out_file.write(gdef.SerializeToString())
    
    # Now we have a TF model ready to be converted to CoreML
    input_tensor_shapes = {'dnn/input_from_feature_columns/input_layer/image/encoded/ToFloat:0':[1,42*90]} # batch size is 1
    coreml_model_file = '{root_dir}/inception_v3.mlmodel'.format(root_dir=model_dir)
    output_tensor_names = ['dnn/head/predictions/probabilities:0']

    # Call the converter. This may take a while
    coreml_model = tfcoreml.convert(
        tf_model_path=frozen_model_file,
        mlmodel_path=coreml_model_file,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names,
        image_input_names = ['dnn/input_from_feature_columns/input_layer/image/encoded/ToFloat:0'],
        red_bias = -1,
        green_bias = -1,
        blue_bias = -1,        
    )        