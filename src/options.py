import argparse


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()

    # Experiment args
    parser.add_argument("--model", type=str, default="omic", help="Model to use")
    parser.add_argument(
        "--task",
        type=str,
        default="multi",
        choices=["multi", "surv", "grad"],
        help="Task to perform",
    )

    # Data args
    parser.add_argument("--use_rna", type=int, default=1, help="Use RNA data")

    # Training args
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument(
        "--w_reg", type=float, default=3e-4, help="L1 regularization weight"
    )

    # Misc
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument(
        "--n_folds", type=int, default=15, choices=range(1, 16), help="CV folds"
    )
    opt = parser.parse_args()
    return opt
