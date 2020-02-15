import argparse
import os
import datetime

import trainer.model as model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--train_data_paths',
        help = 'GCS or local path to training data',
        required = True
    )

    parser.add_argument(
        '--test_data_paths',
        help = 'GCS or local path to testing data',
        required = True
    )

    # Training arguments
    parser.add_argument(
        '--batch_size',
        help = 'Batch size',
        type = int,
        default = 150
    )

    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )

    args = parser.parse_args()

    # Assign model variables to commandline arguments
    model.TRAIN_DATA_DIR = args.train_data_paths
    model.TEST_DATA_DIR = args.test_data_paths
    model.DATA_BATCH_SIZE = args.batch_size
    model.LOGS_DIR = os.path.join(args.output_dir, str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

    # Run the training job
    model.train_and_evaluate()
