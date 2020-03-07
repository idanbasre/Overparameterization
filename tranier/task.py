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

    parser.add_argument(
    '--layer_width',
    help = 'architecture of the linear network: comma delimited widths of the network. i.e.: "2,3,5" is a network with inputs of dim 2, inner layer of dim 3*5, and output of dim 5.',
    required = True
    )

    parser.add_argument(
    '--p',
    help = 'Order of loss function (l_p loss)',
    required = True
    )

    parser.add_argument(
    '--num_epochs',
    help = 'number of epcohs for training',
    required = True
    )

    parser.add_argument(
    '--lr',
    help = 'Gradient Descent learning rate',
    required = True
    )

    args = parser.parse_args()
    layer_width = [int(width) for width in args.layer_width.split(",")]
    p = int(args.p)
    lr = float(args.lr)
    num_epochs = int(args.num_epochs)

    # Run the training job
    model.train_and_evaluate(args.train_data_dir, args.logs_dir, layer_width, p, num_epochs, lr)
