#ibin/bash

TRAIN_DATA_PATH=Small_Data/Train
TEST_DATA_PATH=Small_Data/Test
BATCH_SIZE=10
OUTPUT_DIR=Logs
python -m trainer.task --train_data_paths $TRAIN_DATA_PATH --test_data_paths $TEST_DATA_PATH --batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR

