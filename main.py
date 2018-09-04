import argparse
import sys

sys.path.append('..')
import RDN.rdn as rdn
import RDN.train as train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_global_layers', type=int, default=16, help='-')
    parser.add_argument('--n_local_layers', type=int, default=6, help='-')
    parser.add_argument('--scale', type=int, default=2, help='-')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='-')
    parser.add_argument('--beta_1', type=float, default=0.9, help='-')
    parser.add_argument('--beta_2', type=float, default=0.999, help='-')
    parser.add_argument('--epsilon', type=float, default=1e-08, help='-')
    parser.add_argument('--training_epoch', type=int, default=200, help='-')
    parser.add_argument('--batch_size', type=int, default=128, help='-')
    parser.add_argument('--test_batch_size', type=int, default=5, help='-')
    parser.add_argument('--n_channel', type=int, default=3, help='-')
    args, unknown = parser.parse_known_args()

    RDN = rdn.RDN(args)
    train.training(RDN, args)
