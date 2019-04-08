import os
import sys
import tfcoreml
import tensorflow as tf

from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

if __name__ == "__main__":

    # HELP
    # python2.7 coremltransform output_name model_dir
    # python2.7 coremltransform MeteoML ../data/wind-model/saved-models/1549906046_84

    # Iput parameters        
    output_name = sys.argv[1] + '.mlmodel'
    model_dir = sys.argv[2] 
    tf_model_path = os.path.join(model_dir, 'optimized_model.pb')
    transform_graph_path = '~/Projects/tensorflow-master/bazel-bin/tensorflow/tools/graph_transforms'

    # Prepare graph for conversion
    os.system('{root_path}/transform_graph \
        --in_graph={model_dir}/frozen_model.pb \
        --out_graph={model_dir}/optimized_model.pb \
        --inputs=\'dnn/input_from_feature_columns/input_layer/image/encoded/ToFloat\' \
        --outputs=\'dnn/head/predictions/probabilities\' \
        --transforms=\'strip_unused_nodes(type=float, shape="1,42,90,1") \
            remove_nodes(op=Identity, op=CheckNumerics) \
            fold_constants(ignore_errors=true) \
            fold_batch_norms fold_old_batch_norms\''.format(
                root_path=transform_graph_path,
                model_dir=model_dir,
            ))

    # Load the TF graph definition
    with open(tf_model_path, 'rb') as in_file:
        serialized = in_file.read()

    tf.reset_default_graph()
    original_gdef = tf.GraphDef()
    original_gdef.ParseFromString(serialized)
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(original_gdef, name='')
        ops = graph.get_operations()

    # Strip the JPEG decoder and preprocessing part of TF model    
    input_name = 'dnn/input_from_feature_columns/input_layer/image/encoded/ToFloat'    
    input_node_names = [input_name]
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
    # print(''.join([input_name, ':0']))
    input_tensor_shapes = {''.join([input_name, ':0']) :[1,42,90,1]} # batch size is 1
    coreml_model_file = os.path.join(model_dir, output_name)
    output_tensor_names = ['dnn/head/predictions/probabilities:0']
    
    # Call the converter. This may take a while
    coreml_model = tfcoreml.convert(
        tf_model_path=frozen_model_file,
        mlmodel_path=coreml_model_file,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names,
        image_input_names = [''.join([input_name, ':0'])],
        red_bias = -1,
        green_bias = -1,
        blue_bias = -1,        
    )        