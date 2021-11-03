import os
import warnings
from os import listdir
from os.path import isfile, join
from pathlib import Path
import glob

import cv2
import numpy as np
import tensorflow as tf
from models.VGG16 import vgg_model
from models.depthwise_conv import depthwise_conv_model
from models.zfnet import zfnet_model
from utils import load_augmented_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore tensorflow warnings
warnings.filterwarnings("ignore")

batch_size = 32
img_height = 224
img_width = 224


if __name__ == '__main__':
    test_dir = "../data/chest_xray/chest_xray/test"
    train_dir = "../data/chest_xray/chest_xray/train"
    # val_dir = "../data/chest_xray/chest_xray/val"
    mnist = tf.keras.datasets.mnist

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    # print(training_images[1].shape)
    # print(training_labels)

    # files = [f for f in glob.glob(os.path.join(test_dir, '*'))]
    # print(files)

    X = np.empty(624, dtype=object)
    y = []
    for files in glob.glob(os.path.join(test_dir, '*')):
        print("--------------------------")
        print(files)
        for i in glob.glob(os.path.join(files, '*')):
            if files == test_dir + '/PNEUMONIA':
                X[i] = cv2.imread(join(test_dir + '/PNEUMONIA', files[i]))
                y.append(1)
            if files == test_dir + '/NORMAL':
                X[i] = cv2.imread(join(test_dir + '/NORMAL', files[i]))
                y.append(0)

    print(y)
    print(len(y))
    print("------------------")
    print(X)
    (224,224,1)

    # for n in range(0, len(files)):
    #     images[n] = cv2.imread(join(test_dir, onlyfiles[n]))
    #     print(n)


    # =================================================    =================================================
    # train, valid = load_augmented_data(train_dir, test_dir)
    #
    # input_shape = (224, 224, 1)
    #
    # print(train.shape)
    # model, reduce_lr = zfnet_model(input_shape, 2)
    # # model = vgg_model()
    # # model = depthwise_conv_model()
    #
    # print("Training...")
    # model.fit(train, validation_data=valid, epochs=5, batch_size=5, callbacks=[reduce_lr])
    #
    # val = model.evaluate(valid, steps=200)
    # print("Accuracy: " + str(val)[1])
