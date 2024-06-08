import argparse


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
    parser.add_argument("--mil", type=str, default='PFS', choices=['PFS', 'global', 'local'])
    parser.add_argument("--rna", type=int, default=1, help="Use RNA data")
    parser.add_argument("--attn_pool", type=int, default=0, choices=[0, 1], help="Use attention pooling")
    parser.add_argument("--collate", type=str, default="pad", choices=["pad", "min"], help="Collation method for path data")
    parser.add_argument("--pre_encoded_path", type=int, default=1, help="Use pre-extracted VGG features")
    parser.add_argument("--use_vggnet", type=int, default=1, help="Use VGG or ResNet for path data")

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

    # QBT args
    # parser.add_argument("--n_heads", type=int, default=4)
    # parser.add_argument("--n_queries", type=int, default=16)
    # parser.add_argument("--transformer_layers", type=int, default=1)
    opt = parser.parse_args()

    # Defaults that dynamically align with paper (if not overridden)
    if "omic" in opt.model and opt.task == "surv":
        parser.set_defaults(rna=1)
    if opt.model == "path":
        parser.set_defaults(batch_size=8, lr=0.0005, l1=0)
    if opt.model == "omic":
        parser.set_defaults(batch_size=64, l2=5e-4)
    if opt.task == "grad" or opt.model in ("path", "graph"):
        parser.set_defaults(rna=0)
    if "qbt" in opt.model:
        parser.set_defaults(lr=0.0005, adam_b1=0.5)
    if opt.model in ("path", "graph"):
        parser.set_defaults(l1=0)
    # NOTE: Dont like these settings but they are in the paper, only using for instance MIL
    if opt.model in ("pathomic", "graphomic", "pathgraphomic"):
        if opt.mil == "PFS":
            parser.set_defaults(lr=0.0001, adam_b1=0.5, lr_fix=10, n_epochs=30)
        else:
            parser.set_defaults(lr=0.0005, adam_b1=0.5, n_epochs=60)

    # Sanity checks
    if not opt.use_vggnet and not opt.pre_encoded_path:
        raise ValueError("Must use pre-encoded path features with ResNet")
    if opt.attn_pool and opt.mil == 'PFS':
        raise ValueError("Attention pooling requires MIL")
    if opt.collate == "min" and opt.mil == 'local':
        raise ValueError("Min collation not supported with local MIL")

    opt = parser.parse_args()
    str_opt = str_options(parser, opt)
    print(str_opt)
    return opt, str_opt


# python train.py --model qbt --task grad --folds 1 --use_rna 0  --lr 0.0005


def str_options(parser, opt):
    """Convert options to string."""

    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        default = parser.get_default(k)
        if v != default:
            comment = "\t[default: %s]" % str(default)
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    return message
