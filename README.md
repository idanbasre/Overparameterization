# Overparameterization For Acceleration
Pytorch implementartion and further experiments of the article
[On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization \ Sanjeev Arora, Nadav Cohen, Elad Hazan](https://arxiv.org/abs/1802.06509)

This repo contains a network with N fully-connected layers (N can be changed).
The network is trained with Gradient Descent with a desired learning rate.
The data that is being used is [Gas Sensor Array Drift Dataset at Different Concentrations Data Set](http://archive.ics.uci.edu/ml/datasets/Gas%2BSensor%2BArray%2BDrift%2BDataset%2Bat%2BDifferent%2BConcentrations).

## Training:
In Unix, run the run.sh file.

In windows, run the run.bat file.

Change the following parameters for your need:
* TRAIN_DATA_DIR: Directory of the data.
* LOGS_DIR: Parent Directory where the training logs dir will be create.
* LAYER_WIDTH: Comma seperated width for a fully-connected netwrok. For example, "128,10,20,1" is a netwrok that gets input of dim 128, has inner layer of dim 10*20, another inner layer of dim 20*1, and output dim of 1.
* P: Order of loss function (l_p loss).
* NUM_EPOCHS: Number of epochs for training.
* LR: Gradient Descent learning rate.

## Using a trained network
For every training, in its logs dir, you have the following files:
* net: Pytorch saved parameters for a trained network. Load it with Load method of Pytorch.
* net.csv: csv file with every epoch information.
* net.json:  json file with a python dictionary with every epoch information.
