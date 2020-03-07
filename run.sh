#!bin/bash

TRAIN_DATA_PATH=Data
OUTPUT_DIR=Logs
python -m trainer.task --train_data_paths $TRAIN_DATA_PATH --output_dir $OUTPUT_DIR
