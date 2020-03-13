#!/bin/bash


TRAIN_DATA_DIR=UCIData
LOGS_DIR=Logs
LAYER_WIDTH=128,10,10,1
P=2
NUM_EPOCHS=10
LR=0.001

python -m trainer.task --train_data_dir $TRAIN_DATA_DIR --logs_dir $LOGS_DIR --layer_width $LAYER_WIDTH --p $P --num_epochs $NUM_EPOCHS --lr $LR
