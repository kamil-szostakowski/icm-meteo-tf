# Meteo TF
Meteo TF project is a set of tools which are designed to produce a [CoreML](https://developer.apple.com/documentation/coreml) machine learning model for meteorograms interpretation.

Meteo TF covers all the steps required to create a CoreML model. The toolchain consists of the following steps.

* __Downloader__ => Download of meteorograms.
* __Editor__ => Assigning appropriate features to meteorograms.
* __Builder__ => Build TFRecord files using feature index produced by the Editor.
* __Trainer__ => Training the model usign previously built data set.
* __Transformer__ => Conversion of model in protobuf format into .mlmodel

### Installation
The simplest installation method for Meteo TF is virtualenv. All the tools are written in python 2.7 so the appropriate pip version needs to be used.

```bash
virtualenv icmmeteotf
pip install requirements.txt
```

We need to build tensorflow from source to use graph_transform tool

## Downloader
Downloader is a tool which downloads all available meteorograms form [Meteo.pl](http://meteo.pl) website. Downloader is constraint to several known locations but the list can be extended directly in the code.

__Usage example__
```python
# destination_dir - directory where meteorogram images will be stored

python2.7 setup.py destination_dir
python2.7 setup.py ../data/prediction-images
```

## Editor
Editor is a GUI tool which assists in creation of the dataset for model training. Entire process is manual and requires going step by step through all the images. As a resoult of that process editor produces a json file containing index of all categorized meteorograms together with detected features.

__Supported phenomena__
* Rain
* Snow
* Strong wind
* Clouds

__Usage example__
```python
# input_path - directory where original images are located.
# output_path - path where the index wil be stored

python2.7 editor.py input_path output_path
python2.7 editor.py ../data/training-images ../data/training-set-index.json
```

## Builder
Builder is a script which is responsible for building a set of TFRecord files based on meteorogram images, features index build by editor and a blueprint structure. Builder produces two files, training.TFRecord (_80%_ of training examples) and prediction.TFRecord (_20%_ of training examples).

__Anatomy of blueprint__

Blureprint is a pattern object defining which meteorograms and features should be used to build TFRecord files. It's not mandatory to create a single TFRecord containing training examples for all the features. It's possible to create a number of TFRecords each for a subset of features. All the blueprints are defined in builder_blueprint.py file. They can be customized if required.

__Currently supported blueprints__
- precipitation (_rain, snow, no precipitation_)
- wind (_strong wind, no wind_)
- clouds (_present, no clouds_)
- full (_contains all the above_)

```python
# blueprint_name - name of the blueprint used for TFRecord building.
# input_path - path to the directory where meteorogram images are stored.
# index_path - path to the feature index file.
# intermediate_path - path to the directory where builder's temporary files will be stored.
# output_path - path where the TFRecords file will be located

python2.7 builder blueprint input_path index_path output_path intermediate_path
python2.7 builder.py wind ../data/training-images ../data/training-set-index.json ../data/wind-model/records/ ../data/tmp/intermediate-set
```

## Trainer
Trainer is a script which is responsible for training a machine learning model based on training examples from TFRecord files.
The result of that training is a frozem model stored in protobuf format. All the training details are in the _feature.py_ and in the script itself. That may be decoupled in the future for easier experimentation.

```python
# input_path - path to the directory where TFRecord files are located.
# output_path - path to the directory where the model data will be stored.

python2.7 trainer.py input_path output_path
python2.7 trainer.py ../data/records/ ../data/saved-models
```

## CoreML transformation
Coremltransform is a tool which converts machine learning model in protobuf format to the CoreML format consumable by iOS apps. Some convertion details are hardcoded in the script as well. This may be decoupled in the future for easier experimentation. Two critical pices of information for covertion purpose are name of imput and output layers. [Netron](https://github.com/lutzroeder/netron) can be used to retreive that information from the protobuf file.

```python
# output_name - name of the class which will be imported to Xcode project.
# model_dir - directory where the model files (in protobuf format) are located.

python2.7 coremltransform output_name model_dir
python2.7 coremltransform MeteoML ../data/saved-models/1549906046
```

## References

* [Tensorflow]()
* [Netron](https://github.com/lutzroeder/netron)
* [TF CoreML](https://github.com/tf-coreml/tf-coreml)
* [graph_transform](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md)

