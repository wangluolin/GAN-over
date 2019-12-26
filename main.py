from option import opt
from train import train
from test import test

if __name__ == "__main__":
    if opt.mode == "train":
        train(opt)
    else:
        test(opt)

