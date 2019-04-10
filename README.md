# Meteo TF
Meteo TF project is a set of tools which are designed to produce a [CoreML](https://developer.apple.com/documentation/coreml) machine learning model for meteorograms interpretation.

Meteo TF covers all the steps required to create a CoreML model. The toolchain consists of the following steps.

* Downloader => Download of meteorograms.
* Editor => Assigning appropriate features to meteorograms.
* Builder => Build TFRecord files using feature index produced by the Editor.
* Trainer => Training the model usign previously built data set.
* Transformer => Conversion of model in protobuf format into .mlmodel

### Installation
The simplest installation method for Meteo TF is virtualenv. All the tools are written in python 2.7 so the appropriate pip version needs to be used.

```
virtualenv icmmeteotf
pip install requirements.txt
```

We need to build tensorflow from source to use graph_transform tool
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
https://github.com/tf-coreml/tf-coreml

## Downloader

## Editor
Editor is a GUI tool which assists in creation of the dataset for model training. Entire process is manual and require going step by step through all the images. As a resoult of that process editor produces a json file containing index of all categorized meteorograms together with detected features.

Anatomy of index file
```
{ 
    "sorted_keys": ["2019010112-400-180", "2018091506-461-215"], 
    "values": {
        "2019010112-400-180": "SRWC",
        "2018091506-461-215": "C"
    } 
}
```

```
python2.7 editor.py set_name input_path
python2.7 editor.py training ../data/
```

## Builder

```
python2.7 builder blueprint output_path
python2.7 builder.py wind ../data/wind-model/records/
```

## Trainer

```
python2.7 trainer.py input_path output_path class_name:label ....
python2.7 trainer.py ../data/wind-model/records/ ../data/wind-model/saved-models wind-strong:0 wind-none:1
```

## CoreML transformation

```
python2.7 coremltransform output_name model_dir
python2.7 coremltransform MeteoML ../data/wind-model/saved-models/1549906046_84
```

## References

* Tensorflow
* Netron
