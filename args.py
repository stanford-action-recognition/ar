import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Action Recognition")

    parser.add_argument(
        "--dataset",
        type=str,
        default="HMDB51",
        help="HMDB51, UCF101 (default: HMDB51)",
    )
    parser.add_argument(
        "--dataset_percentage",
        type=float,
        default=1.0,
        help="dataset percentage (default: 1.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="C3D",
        help="C3D，R2Plus1D R3D (default: C3D)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="input batch size for training (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--clip_max_norm",
        type=float,
        default=0.1,
        help="max norm of the gradients (default: 0.1)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="Adam, SGD (default: Adam)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers (default: 0)",
    )

    # C3D only
    parser.add_argument(
        "--c3d_dropout_rate",
        type=float,
        help="C3D dropout rate",
        nargs="?",
        const="",
    )

    parser.add_argument(
        "--c3d_in_channel",
        type=int,
        default=15,
        help="C3D in channel",
    )

    return parser.parse_args()
