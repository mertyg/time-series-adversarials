import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Running Time Series Adversarial experiments.")
    parser.add_argument("--base-dir", default="./results/")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", default="UCR_Adiac")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model", default="MLP"
    )
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--test-batch-size", default=1024, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--max-epochs", default=50, type=int)
    parser.add_argument("--eval-freq", default=5, type=int)
    parser.add_argument("--adversarial-eval", action="store_true")
    parser.add_argument("--attack-config", default="./configs/pgd_inf_0.1.json")
    parser.add_argument("--distance-loss", action="store_true")
    return parser
