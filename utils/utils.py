import os

import cv2
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

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


def load_augmented_data(train_dir, test_dir):
    new = ImageDataGenerator(rescale=1.0 / 255.0,
                             horizontal_flip=True,
                             zoom_range=.2,
                             shear_range=0.2,
                             width_shift_range=0.01,
                             height_shift_range=0.01
                             )
    train = new.flow_from_directory(train_dir, target_size=(224, 224), class_mode='binary', color_mode='grayscale',
                                    batch_size=32)
    valid = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(test_dir, target_size=(224, 224),
                                                                        class_mode='binary',
                                                                        color_mode='grayscale', batch_size=32)
    return train, valid



def plot_img(train):
    l = []
    for i in train:
        if i[1] == 0:
            l.append("Pneumonia")
        else:
            l.append("Normal")
    sns.countplot(l)
    plt.figure(figsize=(5, 5))
    plt.imshow(train[0][0], cmap='viridis')
    plt.title(labels[train[0][1]])
    plt.show()
    plt.figure(figsize=(5, 5))
    plt.imshow(train[-1][0], cmap='viridis')
    plt.title(labels[train[-1][1]])
    plt.show()


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
