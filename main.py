import os
import warnings

import tensorflow as tf

from utils import load_img

tf.random.set_seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore tensorflow warnings
warnings.filterwarnings("ignore")


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

if __name__ == '__main__':
    path = 'data/chest_xray/chest_xray/'
    train, val = load_img("data/chest_xray/chest_xray/train", "data/chest_xray/chest_xray/val")

    print(train)




    # convnet()
