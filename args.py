import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Action Recognition")
    parser.add_argument(
        "--dataset",
        type=str,
        default="hmdb51",
        help="hmdb51, ucf101 (default: hmdb51)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="C3D",
        help="C3D， R2Plus1D， R3D (default: C3D)",
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
    return parser.parse_args()
