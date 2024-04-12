import argparse
from train import train

if __name__ == "__main__":
    # Paratemter selection
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--name", type=str, default="YaleB")
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--emblem_size", default=128, type=int)
    parser.add_argument("--lambda1", default=1e+0, type=float)
    parser.add_argument("--lambda2", default=1e+3, type=float)
    parser.add_argument("--neighbor", default=5, type=int)
    parser.add_argument('--update_interval', default=6, type=int)
    parser.add_argument("--max_epoch", default=100, type=int)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument("--save", type=bool, default=False)
    args = parser.parse_args()
    train(args)