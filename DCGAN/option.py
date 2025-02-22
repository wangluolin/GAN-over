import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--imageSize', type=int, default=96)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--pth', default='pth/', help='folder to save model checkpoints')
parser.add_argument('--data_path', default='data/', help='folder to train data')
parser.add_argument('--test_path', default='test_imgs/', help='folder to test output data')
parser.add_argument('--outf', default='imgs/', help='folder to output images')
parser.add_argument('--mode', default='test', help='train or test' )
opt = parser.parse_args()