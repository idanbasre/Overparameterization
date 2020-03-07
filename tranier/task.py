import argparse
import os
import datetime

import trainer.model as model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--train_data_dir',
        help = 'GCS or local dir to training data',
        required = True
    )

    parser.add_argument(
        '--logs_dir',
        help = 'GCS/local dir to write checkpoints and export models',
        required = True
    )

    args = parser.parse_args()

    # Run the training job
    model.train_and_evaluate(args.train_data_dir, args.logs_dir)
