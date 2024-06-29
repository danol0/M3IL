import argparse


class CustomNamespace(argparse.Namespace):
    """Custom namespace class for nice printing."""

    def __str__(self):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(self).items()):
            comment = ""
            if 'qbt' in k and self.qbt:
                message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
            elif 'qbt' not in k:
                message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        return message


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()

    # Run settings
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--dry_run", type=int, default=0, help="No save or log")
    parser.add_argument("--folds", type=int, default=15, choices=range(1, 16))
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--resume", type=int, default=0, help="Don't overwrite existing ckpts")

    # Experiment args
    parser.add_argument("--model", type=str, default="omic")
    parser.add_argument("--task", type=str, default="grad", choices=["multi", "surv", "grad"])
    parser.add_argument("--mil", type=str, default="PFS", choices=["PFS", "global", "local"])
    parser.add_argument("--rna", type=int, default=0, help="Use RNA data")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "attn", "LSE"], help="MIL pooling method")
    parser.add_argument("--collate", type=str, default="pad", choices=["pad", "min"], help="Collation method for path data")
    parser.add_argument("--pre_encoded_path", type=int, default=1, help="Use pre-extracted VGG features")

    # Training args
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--lr_fix", type=int, default=0, help="Epochs before lr decay")
    parser.add_argument("--l1", type=float, default=3e-4, help="L1 weight")
    parser.add_argument("--l2", type=float, default=4e-4, help="L2 weight")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--adam_b1", type=float, default=0.9, help="Adam momentum")
    parser.add_argument("--unfreeze_unimodal", type=int, default=5, help="Epoch to unfreeze")

    # QBT
    parser.add_argument("--qbt", type=int, default=0, help="Use QBT model")
    parser.add_argument("--qbt_queries", type=int, default=32, help="Number of queries")
    parser.add_argument("--qbt_dim", type=int, default=16, help="Query dimension")
    parser.add_argument("--qbt_layers", type=int, default=3, help="Number of QBT update layers")
    parser.add_argument("--qbt_heads", type=int, default=4, help="Number of QBT attention heads")

    opt = parser.parse_args(namespace=CustomNamespace())

    # Dynamic defaults
    if "omic" in opt.model and opt.task == "surv":
        parser.set_defaults(rna=1)
    if opt.model == "path":
        parser.set_defaults(batch_size=8, lr=0.0005, l1=0, pre_encoded_path=0)
    if opt.model == "omic":
        parser.set_defaults(batch_size=64, l2=5e-4)
    if opt.model in ("path", "graph"):
        parser.set_defaults(l1=0)
    if opt.model.split('_')[0] in ("pathomic", "graphomic", "pathgraphomic"):
        parser.set_defaults(n_epochs=30, lr_fix=10)
        if opt.mil == "PFS":
            parser.set_defaults(lr=0.0001)
        else:
            parser.set_defaults(lr=0.0005)
    if opt.qbt:
        parser.set_defaults(mil="global", n_epochs=30, lr_fix=10, lr=0.0001, l1=0.0005, unfreeze_unimodal=10)

    # Sanity checks
    if opt.collate == "min" and opt.mil == "local":
        raise ValueError("Min collation not supported with local MIL")
    if opt.model in ("path", "omic") and opt.mil != "PFS":
        # Path is treated as a feature extractor and only supports PFS
        # There is only one omic datapoint per patient so no need for MIL
        raise ValueError("MIL not supported for unimodal path/omic models")
    if opt.l1 > 0.0 and opt.model in ("path", "graph"):
        print("Note: L1 is only enabled for the omic network (or subnetwork in MM models)")
    if (opt.pool == "LSE" and opt.mil in ("PFS")) or (opt.pool == "attn" and opt.mil in ("local", "PFS")):
        raise ValueError("Pooling and MIL strategies are incompatible")
    if opt.qbt and opt.mil != "global":
        raise ValueError("QBT only supports global MIL")

    opt = parser.parse_args(namespace=CustomNamespace())
    return opt
