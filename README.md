# Mahjong Connect and Facial Expression Recognition

A course project for Artificial Intelligence, Autumn 2021 @ Department of Automation, Tsinghua

Lecturer: [Rui Jiang](https://scholar.google.com/citations?user=LkF_AbYAAAAJ&hl=en-US)

## [Mahjong Connect](./MahjongConnect/)

A Automated Solver for 'Mahjong Connect', with GUI.

Download [llkui.exe](https://cloud.tsinghua.edu.cn/f/22e7dd3eb14549ecb8ed/) and put into 'MahjongConnect' folder to execute.

<img src="./MahjongConnect/pic/ui_mainwindow.PNG" width="600">

## [Facial Expression Recognition](./FacialExpression/)

**File List:**

| Dir   | FileName                            | Description (PT refers to Pre-training)                |
| ----- | ----------------------------------- | ------------------------------------------------------ |
| code  | LeNet.ipynb                         | Original data, LeNet (Question 1)                      |
|       | vgg9.ipynb                          | Original data, VGG-9 (Question 1)                      |
|       | ResNet-11.ipynb                     | Original data, ResNet-11 (Question 1)                  |
|       | vgg9-DataEnhance.ipynb              | Enhanced data, VGG-9 (Question 2)                      |
|       | vgg11aug_balance_pretrain.ipynb     | Enhanced data, PT VGG-11                               |
|       | ResNet18aug_balance_pretrain.ipynb  | Enhanced data, PT ResNet-18                            |
|       | ResNet18aug_balance_pretrain2.ipynb | Enhanced data, PT ResNet-18                            |
|       | TestTime.ipynb                      | Test the running speed of each model's prediction mode |
|       | myui.ui                             | QT file for the graphical interface                    |
|       | myui.py                             | Runnable final program (Question 3)                    |
| model | vgg9net.pkl                         | Original data, VGG-9 model, Acc58.4%                   |
|       | vgg9net_DataEnhance.pkl             | Enhanced data, VGG-9 model, Acc58.8%                   |
|       | finetune_vgg11net.pkl               | Enhanced data, PT VGG-11, Acc62.1%                     |
|       | finetune_resnet18.pkl               | Enhanced data, PT ResNet-18, Acc59.1%                  |
| data  | data.csv  ;  *.pt                   | Original data; Processed tensor files                  |
| media | N/A                                 | Images/videos for testing the GUI program              |



**Usage:**

- First, perform **model training and testing**, there are two methods:

1. Download the entire data folder, run the .ipynb code in the code folder, at this time directly read data from the data folder;

2. Change the readtensor parameter in the .ipynb code to False, only put data.csv in the data folder, no need to download .pt files, at this time re-process data from data.csv (computationally time-consuming);



- Then perform **data prediction testing**, use the GUI program (Question 3):

1. Install the following packages in the python environment: pyqt5, opencv-python, torch, torchvision

2. Put the trained vgg9net.pkl in the model folder, then run the myui.py in the code folder

[Download Data](https://cloud.tsinghua.edu.cn/f/037773e21aae4859be45/)

