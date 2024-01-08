import argparse
from fgan_train_torch import training_pipeline

parser = argparse.ArgumentParser('Train your Fence GAN')

# Training hyperparameter
args = parser.add_argument('--dataset', type=str,
                           default='mnist', help='mnist | cifar10')
args = parser.add_argument('--ano_class', type=int, default=2, help='1 anomaly class')
args = parser.add_argument('--epochs', type=int, default=100,
                           help='number of epochs to train')

# FenceGAN hyperparameter
args = parser.add_argument('--beta', type=float, default=30, help='beta')
args = parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
args = parser.add_argument('--alpha', type=float, default=0.5, help='alpha')

# Other hyperparameters
args = parser.add_argument('--id', '-id', type=str,
                           default='run1', help='unique run id')
args = parser.add_argument('--batch_size', type=int, default=200, help='')
args = parser.add_argument('--pretrain', type=int, default=15,
                           help='number of pretrain epoch')
args = parser.add_argument('--d_l2', type=float, default=0,
                           help='L2 Regularizer for Discriminator')
args = parser.add_argument('--d_lr', type=float, default=1e-5,
                           help='learning_rate of discriminator')
args = parser.add_argument('--g_lr', type=float, default=2e-5,
                           help='learning rate of generator')
args = parser.add_argument('--v_freq', type=int, default=4,
                           help='epoch frequency to evaluate performance')
args = parser.add_argument('--seed', type=int, default=0,
                           help='numpy and tensorflow seed')
args = parser.add_argument('--evaluation', type=str,
                           default='auprc', help="'auprc' or 'auroc'")
args = parser.add_argument('--latent_dim', type=int, default=200,
                           help='Latent dimension of Gaussian noise input to Generator')
args = parser.add_argument('--ol', '-ol', action='store_true',
                           help='Include Outline Loss')
args = parser.add_argument('--kappa', '-k', type=float, default=0.0015,
                           help='weighting hyperparameter for outline loss. Default \
                               0.0015 for MNIST and 0.2 for CIFAR-10')
args = parser.add_argument('--omega', '-o', type=float, default=945,
                           help='closeness hyperparameter for outline loss. Default \
                               945 for MNIST and 36 for CIFAR-10')
args = parser.add_argument('--save_model', '-sav', action='store_false',
                           help='omit saving model')
args = parser.add_argument('--progress_bar', '-prg', action='store_false',
                           help='omit progress bar')
args = parser.add_argument('--bm', '-bm', type=float, default=1,
                           help='weighting hyperparameter for encirclement loss')
args = parser.add_argument('--small', '-s', action='store_true',
                           help='10% dataset size')
args = parser.add_argument('--verif', '-vf', action='store_true',
                           help='verbose for outline loss')

args = parser.parse_args()

training_pipeline(args)
