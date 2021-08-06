import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--mnist_path', type=str, default=r'./mnist.npz')
parser.add_argument('--model_save_dir', type=str, default=r'./model')

parser.add_argument('--img_size', type=int, default=(28, 28))
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_classes', type=int, default=10)

parser.add_argument('--model_path', type=str, default=r'../mnistModel/mnistModel.{epoch:03d}--{acc:.4f}.h5')


cfg = parser.parse_args()

if __name__ == '__main__':
    print(*cfg.img_size)





















