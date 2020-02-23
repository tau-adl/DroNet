#!bin/bash

TRAIN_DATA_PATH=../Medium_Data/Train
TEST_DATA_PATH=../Medium_Data/Test
BATCH_SIZE=200
OUTPUT_DIR=Logs/DroNet
python -m trainer.task --train_data_paths $TRAIN_DATA_PATH --test_data_paths $TEST_DATA_PATH --batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR

