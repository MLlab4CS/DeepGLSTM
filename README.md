# DeepGLSTM: Deep Graph Convolutional Network and LSTM based approach for predicting drug-target binding affinity
# Quick Links
1. [Model Architecture](#Model-Architecture)
2. [Preparation](#prepration)
   1. [Environment Setup](#env-setup)
   2. [Dataset description](#dataset)
3. [Quick Start](#start)
   1. [Model Training](#model-tra)
   2. [Inference on Pretrained Model](#Inf-pre)
4. [Pretrained Models and Dataset](#premod-data)
   1. [Pretrained Models download links](#P-down)
   2. [Dataset download links](#data-down)
5. [Statistics](#stats)
6. [Citation](#cite)


## Model Architecture <a name="Model-Architecture"></a>
![alt text](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/architecture.jpg "DeepGLSTM")

## Preparation <a name="prepration"></a>
### Environment Setup <a name="env-setup"></a>
The dependency pakages can be installed using the command
```python
pip install -r requirements.txt
```
### Dataset description <a name="dataset"></a>
In our experiment we use Davis, Kiba, DTC, Metz, ToxCast, Stitch datasets respectively. The statistics of the datasets are shown below
