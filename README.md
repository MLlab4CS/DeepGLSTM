# DeepGLSTM: Deep Graph Convolutional Network and LSTM based approach for predicting drug-target binding affinity
# Quick Links
1. [Model Architecture](#Model-Architecture)
2. [Preparation](#prepration)
   1. [Environment Setup](#env-setup)
   2. [Dataset description](#dataset)
3. [Quick Start](#start)
   1. [Create Dataset](#create-dataset)
   2. [Model Training](#model-tra)
   3. [Inference on Pretrained Model](#Inf-pre)
4. [Pretrained Models and Dataset](#premod-data)
   1. [Pretrained Models download links](#P-down)
   2. [Dataset download links](#data-down)
5. [Statistics](#stats)
6. [Citation](#cite)


## Model Architecture <a name="Model-Architecture"></a>
DeepGLSTM:
![alt text](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/architecture.jpg "DeepGLSTM")

## Preparation <a name="prepration"></a>
### Environment Setup <a name="env-setup"></a>
The dependency pakages can be installed using the command
```python
pip install -r requirements.txt
```
### Dataset description <a name="dataset"></a>
In our experiment we use Davis, Kiba, DTC, Metz, ToxCast, Stitch datasets respectively.

Dataset Statistics:

![alt text](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/dataset_statistics.png "Dataset statistics")

## Quick Start <a name="model-tra"></a>
### Create Dataset <a name="create-dataset"></a>
Firstly, run the script below to create Pytorch_Geometric file. The file will be created in processed directory in data directory.
```python
python3 data_creation.py 
```
Default values of argument parser are set for davis dataset.
### Model Training  <a name="model-tra"></a>
Run the following script to train the model.
```python
python3 training.py 
```
Default values of argument parser are set for davis dataset.
### Inference on Pretrained Model  <a name="Inf-pre"></a>
Run the following script to train the model.
```python
python3 inference.py 
```
Default values of argument parser are set for davis dataset.

## Pretrained Models and Dataset <a name="premod-data"></a>
### Pretrained Models download links <a name="P-down"></a>
| Dataset   | Model download link |
| --------- | :------------------:|
| Davis     | [Link](https://drive.google.com/file/d/1-lzd2Hq5bidsdJI8gGvfIducHDwL_PLd/view?usp=sharing) |
