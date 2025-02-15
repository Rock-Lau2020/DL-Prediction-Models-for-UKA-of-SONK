# Deep learning-based prediction model with contrastive learning for identifying SONK patients suitable for unicompartmental knee arthroplasty


[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://opensource.org/licenses/Apache-2.0)
<!-- [![DOI]()]()-->


## Introduction
This is the official repository of 

**Deep learning-based prediction model with contrastive learning for identifying SONK patients suitable for unicompartmental knee arthroplasty** 

by
*Hongzhi Liu, Xiaoyao Wang, Yicong Chen, Xinqiu Song, Fuzhou Du, and Hongmei Zhang*

Please note that this project only provides the code for the classifier part, while the ROI retrieval is implemented using YOLOv5 code. For that part of the code, please refer to the YOLOv5 code.

## Installation Guide:
The code is based on Python 3.6.8

1. Download the repository
```bash
git clone https://github.com/.../...
```

2. Install requested libarary.
```bash
cd sonkModel
pip install -r requirements.txt
```
Typically, it will take few minutes to complete the installation.


## Run
1. The model of our paper is defined in ```model.py```, the datasets is defined in ```dataset.py```.
2. Train the model through ```train.py``` and test through ```test.py```.
#### Run the following command for training:
```bash
python train.py
```

#### Run the following command for testing:
```bash
python test.py
```
Please note that you may need to adjust the paths and content of the data set file folder corresponding to the train.py and test.py files for these files.

## License & Citation

This project is covered under the **Apache 2.0 License**.
