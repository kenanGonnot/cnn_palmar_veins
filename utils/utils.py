import gc
import glob
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150


def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
                # data.append([img_arr, class_num])
            except Exception as e:
                print(e)
    return data
    # return np.array(data)


def load_augmented_data(train_dir, test_dir, val_dir):
    new = ImageDataGenerator(rescale=1.0 / 255.0,
                             horizontal_flip=True,
                             zoom_range=.2,
                             shear_range=0.2,
                             width_shift_range=0.01,
                             height_shift_range=0.01
                             )
    train = new.flow_from_directory(train_dir, target_size=(224, 224), class_mode='binary', color_mode='grayscale',
                                    batch_size=32)
    test = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(test_dir, target_size=(224, 224),
                                                                       class_mode='binary',
                                                                       color_mode='grayscale', batch_size=32)
    valid = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(val_dir, target_size=(224, 224),
                                                                        class_mode='binary',
                                                                        color_mode='grayscale', batch_size=32)
    return train, test, valid


def load_images(img_dir, xdim, ydim, nmax=5000):
    label = 0
    label_names = []
    X = []
    y = []
    for dirname in os.listdir(img_dir):
        print(dirname)
        label_names.append(dirname)
        data_path = os.path.join(img_dir + "/" + dirname, '*g')
        files = glob.glob(data_path)
        n = 0
        for f1 in files:
            if n > nmax: break
            img = cv2.imread(f1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (xdim, ydim))
            X.append(np.array(img))
            y.append(label)
            n = n + 1
        print(n, ' images lues')
        label = label + 1
    X = np.array(X)
    y = np.array(y)
    gc.collect()
    return X, y, label, label_names


def plot_img(X, y, Classes):
    plt.figure(figsize=(10, 20))
    for i in range(0, 49):
        plt.subplot(10, 5, i + 1)
        j = random.randint(0, len(X))
        plt.axis('off')
        plt.imshow(X[j])
        plt.title(Classes[y[j]])
    plt.show()


def plot_scores(train):
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    plt.show()

def previsualization_normalization_data(X_train, X_test, X_val, y_train, y_test, y_val, Nombre_classes, Classes):
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
    return X_train, X_test, X_val, y_train, y_test, y_val


def load_img(data_dir_train, data_dir_val):
    batch_size = 32
    img_height = 267
    img_width = 267
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir_train,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir_val,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return train_ds, val_ds
