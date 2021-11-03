import os
import warnings

import tensorflow as tf

from models.VGG16 import vgg_model
from models.depthwise_conv import depthwise_conv_model
from models.zfnet import zfnet_model
from utils.utils import load_img, load_augmented_data

tf.random.set_seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore tensorflow warnings
warnings.filterwarnings("ignore")


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

if __name__ == '__main__':
    train_dir = "data/chest_xray/chest_xray/train"
    test_dir = "data/chest_xray/chest_xray/test"
    input_shape = (224, 224, 1)
    classes = 1

    train, valid = load_augmented_data(train_dir, test_dir)
    # print(train)

    # model = vgg_model()
    # model = depthwise_conv_model()
    model, reduce_lr = zfnet_model(input_shape, classes)
    #
    model.fit(train, validation_data=valid, epochs=5, batch_size=5, callbacks=[reduce_lr])
    # model.fit(train, validation_data=valid, epochs=5, batch_size=5)
    #
    val = model.evaluate(valid, steps=200)
    print("Accuracy: " + str(val)[1])





    # convnet()
