import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Action Recognition")

    parser.add_argument(
        "--is_toy",
        type=bool,
        default=False,
        help="If is_toy, then the number of train iters is reduced to train_toy_size, the number of val iters is reduced to val_toy_size.",
    )
    parser.add_argument(
        "--train_toy_size",
        type=int,
        default=100,
        help="If is_toy, then the number of train iters is reduced to train_toy_size.",
    )
    parser.add_argument(
        "--val_toy_size",
        type=int,
        default=10,
        help="If is_toy, then the number of val iters is reduced to val_toy_size.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HMDB51",
        help="HMDB51, UCF101 (default: HMDB51)",
    )
    parser.add_argument(
        "--streams",
        type=str,
        default="rgb,flow",
        help="Comma separated list of streams. Available streams are rgb, flow. All streams are on HMDB51 dataset.",
    )
    parser.add_argument(
        "--use_pretrained",
        type=bool,
        default=False,
        help="Whether to load pretrained models.",
    )
    parser.add_argument(
        "--freeze_stream_models",
        type=bool,
        default=False,
        help="Whether to change the parameters of stream models during training.",
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
        default="R3D",
        help="C3D, R2Plus1D, R2Plus1D_BERT, R3D, R3D_BERT (default: R3D)",
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
        default=10000,
        help="number of epochs to train (default: 10000)",
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
        default=0.2,
        help="C3D dropout rate",
    )

    parser.add_argument(
        "--c3d_in_channel",
        type=int,
        default=3,
        help="C3D in channel",
    )

    parser.add_argument(
        "--clip_len",
        type=int,
        default=32,
        help="Temporal dimension",
    )

    return parser.parse_args()
