import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Running Time Series Adversarial experiments.")
    parser.add_argument("--dataset", default="UCR_Adiac")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model", default="MLP"
    )
    parser.add_argument(
        "--optimizer",
        default="Adam_0.01",
        help="Please pass this as OptimizerName_LR_Param1_Param2..",
    )
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--test-batch-size", default=1024, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--max-epochs", default=500, type=int)
    parser.add_argument("--eval-freq", default=1, type=int)
    parser.add_argument("--results-folder", default="./results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument("--transform-weight", default=0.01)
    parser.add_argument("--resume", default="", help="Location of the checkpoint file to resume training.")

    return parser
