# Project title

Deep Learning and Feature Visualization

# Prerequisites

## Installing

python==3.7.1

matplotlib==3.0.1

seaborn==0.9.0

sklearn==0.20.0

pytorch==0.4.1

# Folder

## Preparation

You should prepare empty folders: datasets, datasets/MNIST, Model

## Generation

The MNIST will be downloaded from internet and store in datasets/MNIST

The models for network will be stored in Model folder

# Command

python main.py [-h] -path PATH [-load {y,n}] -dataset {MNIST,CUB,CIFAR} -pattern {train,test} -net {AlexNet,VGG16,VGG19,ResNet}

note: the path must be in Model folder which means that you should input Model/