import os
import warnings

import tensorflow as tf
from keras.utils.np_utils import to_categorical

from models.VGG16 import vgg_model
from models.depthwise_conv import depthwise_conv_model
from models.old_zfnet_model import old_zfnet_model
from models.zfnet import zfnet_model
from utils.utils import load_img, load_augmented_data, load_images, plot_img, plot_scores

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

    print("\n--------------------------------")
    print("X_train shape : {}".format(X_train.shape))
    print("--------------------------------")
    print("y_train shape : {}".format(y_train.shape))
    print("--------------------------------")
    print("Nombre_classes = %d" % Nombre_classes)
    print("--------------------------------")
    print("Classes:", Classes)
    print("--------------------------------")

    plot_img(X_train, y_train, Classes)

    X_train = X_train / 255.
    X_test = X_test / 255.
    X_val = X_val / 255.
    # X_train = X_train.reshape(X_train.shape[0], 224, 224, 1)  # add dimension => 4D tensor
    # X_test = X_test.reshape(X_test.shape[0], 224, 224, 1)
    # X_val = X_val.reshape(X_val.shape[0], 224, 224, 1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    # train, test, valid = load_augmented_data(train_dir, test_dir, val_dir)

    # model = vgg_model()
    # model = depthwise_conv_model()
    model, reduce_lr = zfnet_model(input_shape, Nombre_classes)
    # model = old_zfnet_model(input_shape, classes)
    # model.fit(train, validation_data=valid, epochs=4, batch_size=5, callbacks=[reduce_lr])
    # model.fit(train, validation_data=valid, epochs=1, batch_size=32)
    train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[reduce_lr])
    val = model.evaluate(X_test, y_test)
    # pred = model.predict(test)
    print("Score : %.2f%%" % (val[1]*100))
    # print("Predictions: " + str(pred))

    plot_scores(train)





