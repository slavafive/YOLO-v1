# YOLO-v1

Implementation of the YOLO algorithm (version 1) paper in PyTorch: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf).

## Description
The YOLO model was trained on [Pascal VOC dataset](https://www.kaggle.com/datasets/aladdinpersson/pascalvoc-yolo).
The hyperparameters of the model can be found in the `config.yaml` file.

## Project structure

* `src/`
    * `metrics/`
        * `cbow.py` - implemented CBOW model.
        * `iou.py` - implemented Skip-Gram model.
        * `mean_average_precision.py` - contains common function used for models.
    * `model/`
        * `model.py` - YOLO architecture.
    * `utils`
        * `utils_file` - utilities for loading and saving files.
        * `utils_list.py` - utilities for operating on lists.
    * `dataset.py` - implementation of the PyTorch Dataset in Pascal VOC format.
    * `loss.py` - implementation of the loss function used for training YOLO.
    * `main.py` - full training pipeline.
    * `non_max_suppression.py` - implementation of the Non-maximum Suppression used during the model inference.
    * `train.py` - class implementation for training a model.
* `test/` - collection of unit tests
* `config.yaml` - config file with the main parameters of datasets and the model.

## Usage
1. Clone the repository into the `yolo` folder:
```
git clone https://github.com/slavafive/YOLO-v1.git yolo
```
2. Download the [Pascal VOC dataset](https://www.kaggle.com/datasets/aladdinpersson/pascalvoc-yolo). Save it in the `yolo` folder as the `data` directory.
3. Run the training pipeline:
```
python yolo/src/main.py --config yolo/config.yaml
```
4. After the training, the model and its checkpoints will be saved in the directory specified in the `model_directory` (`artifacts` by default) attribute of the `config.yaml` file.

## Resources
A part of the source code and unit tests were used from [Aladdin Persson repository](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection).