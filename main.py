import os
import warnings

from keras import Sequential
from keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
from models.VGG16 import vgg_model
from models.depthwise_conv import depthwise_conv_model
from models.old_zfnet_model import old_zfnet_model
from models.zfnet import zfnet_model
from utils.utils import load_img, load_augmented_data, load_images, plot_img, plot_scores, \
    previsualization_normalization_data
from tensorflow.keras.applications import vgg19

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
    val_dir = "data/chest_xray/chest_xray/val"
    input_shape = (224, 224, 3)
    xdim = 224
    ydim = 224

    X_train, y_train, Nombre_classes, Classes = load_images(train_dir, xdim, ydim, 1000)
    X_test, y_test, Nombre_classes, Classes = load_images(test_dir, xdim, ydim, 1000)
    X_val, y_val, Nombre_classes, Classes = load_images(val_dir, xdim, ydim, 1000)

    X_train, X_test, X_val, y_train, y_test, y_val = previsualization_normalization_data(X_train, X_test, X_val, y_train, y_test, y_val, Nombre_classes, Classes)

    # train, test, valid = load_augmented_data(train_dir, test_dir, val_dir)

    # model = vgg_model()
    # model = depthwise_conv_model()
    # model, reduce_lr = zfnet_model(input_shape, Nombre_classes)
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg16.trainable = False
    model = Sequential()
    model.add(vgg16)
    model.add(Flatten())
    model.add(Dense(Nombre_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    # model = old_zfnet_model(input_shape, classes)
    # model.fit(train, validation_data=valid, epochs=4, batch_size=5, callbacks=[reduce_lr])
    # model.fit(train, validation_data=valid, epochs=1, batch_size=32)
    train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30)
    val = model.evaluate(X_test, y_test)
    print("Score : %.2f%%" % (val[1]*100))

    plot_scores(train)

    predict_x = model.predict(X_test)
    y_cnn = np.argmax(predict_x, axis=1)

    plt.figure(figsize=(15, 25))
    n_test = X_test.shape[0]
    i = 1
    for j in range(len(X_test)):
        if (y_cnn[j] != y_test[j].argmax(axis=-1)) & (i < 10):
            plt.subplot(10, 5, i)
            plt.axis('off')
            plt.imshow(X_test[j])
            plt.title('%s / %s' % (Classes[y_cnn[j]], Classes[y_test[j].argmax(axis=-1)]))
            i += 1


    # skmodel = KerasClassifier(build_fn=create_model)


