# DeepGLSTM: Deep Graph Convolutional Network and LSTM based approach for predicting drug-target binding affinity
# Quick Links
1. [Abstract](#task)
2. [Model Architecture](#Model-Architecture)
3. [Preparation](#prepration)
   1. [Environment Setup](#env-setup)
   2. [Dataset description](#dataset)
4. [Quick Start](#start)
   1. [Create Dataset](#create-dataset)
   2. [Model Training](#model-tra)
   3. [Inference on Pretrained Model](#Inf-pre)
5. [Pretrained Models and Dataset](#premod-data)
   1. [Pretrained Models download links](#P-down)
   2. [Dataset download links](#data-down)
6. [Model Performance Stats](#stats)
7. [Case studies on SARS-CoV-2 viral proteins](#case)
8. [Citation](#cite)

## Abstract <a name="task"></a>
Development of new drugs is an expensive  and time-consuming process. Due to the world-wide SARS-CoV-2 outbreak, it is essential that new drugs for SARS-CoV-2 are developed as soon as possible. Drug repurposing techniques can reduce the time span needed to develop new drugs by probing the list of existing FDA-approved drugs and their properties to reuse them for combating the new disease. We propose a novel architecture DeepGLSTM, which is a Graph Convolutional network and LSTM based method that predicts binding affinity values  between the FDA-approved drugs and the viral proteins of SARS-CoV-2. Our proposed model has been trained on Davis, KIBA (Kinase Inhibitor Bioactivity), DTC (Drug Target Commons), Metz, ToxCast and STITCH datasets. We use our novel architecture to predict a Combined Score (calculated using Davis and KIBA score) of 2,304 FDA-approved drugs against 5 viral proteins. On the basis of the Combined Score, we prepare a list of the top-18 drugs with the highest binding affinity for 5 viral proteins present in SARS-CoV-2. Subsequently, this list may be used for the creation of new useful drugs. For more details please visit our [work](https://arxiv.org/abs/2201.06872v1).


## Model Architecture <a name="Model-Architecture"></a>
![alt text](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/architecture.jpg "DeepGLSTM")

## Preparation <a name="prepration"></a>
### Environment Setup <a name="env-setup"></a>
The dependency pakages can be installed using the command.
```python
pip install -r requirements.txt
```
### Dataset description <a name="dataset"></a>
In our experiment we use Davis, Kiba, DTC, Metz, ToxCast, Stitch datasets respectively.

Dataset Statistics:

![alt text](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/dataset_statistics.png "Dataset statistics")

## Quick Start <a name="model-tra"></a>
### Create Dataset <a name="create-dataset"></a>
Firstly, run the script below to create Pytorch_Geometric file. The file will be created in processed folder in data folder.
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
| Kiba      | [Link](https://drive.google.com/file/d/1buwSFWxmyBOLSdJ9BiMOa8E-GvMGJnar/view?usp=sharing) |
| DTC       | [Link](https://drive.google.com/file/d/1Pam_irCkpKsvNGIdJM8rC9r79u6o5Q7t/view?usp=sharing) |
| Metz      | [Link](https://drive.google.com/file/d/1X4qhc-9zmwiGPB_83NFgTiA-cOUStQeJ/view?usp=sharing) |
| ToxCast   | [Link](https://drive.google.com/file/d/1r4y-a7rhfcYjvWLBwRqW5ckfeewHNH_9/view?usp=sharing) |
| Stitch    | [Link](https://drive.google.com/file/d/1JwIhSrSRUR1CEEZc6kIlNiphPHa47_x9/view?usp=sharing) |

Download models from the above table for particular dataset and store in the pretrained_model folder.

### Dataset download links <a name="P-down"></a>
| Dataset   | Dataset download links |
| --------- | :------------------:|
| Davis     |[Link](https://drive.google.com/drive/folders/17ZmLlkUBqz8f3nVJQebLDDo90JYrelA2?usp=sharing)|
| Kiba      |[Link](https://drive.google.com/drive/folders/1vqRSVzwF97UISUZDlF2oeg0K3Rw7jVkS?usp=sharing)|
| DTC       |[Link](https://drive.google.com/drive/folders/1or9YSjw-LXIUy4ch8ZtAmf7Wl5IEDMjn?usp=sharing)|
| Metz      |[Link](https://drive.google.com/drive/folders/1LZI1GJzsXvLiOKlJVzsawrITQ9fuQl27?usp=sharing)|
| ToxCast   |[Link](https://drive.google.com/drive/folders/1L9i8h5jMaIuzF1rXBaJFXMqlWVXvSoDo?usp=sharing)|
| Stitch    |[Link](https://drive.google.com/drive/folders/1tC8gHn-sIINuEiGbgY8zSMYIJydgmkzE?usp=sharing)|

Download dataset from the above table for particular data and store in the data folder. For each folder in the link there are two csv file train and test.

## Model Performance Stats <a name="stats"></a>

![alt text](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/Full_fig%20.jpg "Full_fig")

Plots showing DeepGLSTM versus measured binding affinity values for the (a)  Davis dataset (b) KIBA dataset (c) DTC dataset (d) Metz dataset (e) ToxCast dataset (f) STITCH dataset. In figure Coef_V is Pearson correlation coefficient.

## Case studies on SARS-CoV-2 viral proteins <a name="case"></a>
![alt text](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/Sup_table.jpeg "Sup_1")
![alt text](https://github.com/MLlab4CS/DeepGLSTM/blob/main/images/sup_table2.jpeg "Sup_2")

## Citation  <a name="cite"></a>
Please cite our paper if it's helpful to you in your research.

```bibtext 
@misc{mukherjee2022deep,
      title={Deep Graph Convolutional Network and LSTM based approach for predicting drug-target binding affinity}, 
      author={Shrimon Mukherjee and Madhusudan Ghosh and Partha Basuchowdhuri},
      year={2022},
      eprint={2201.06872},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
