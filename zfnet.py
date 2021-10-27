import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns


def convnet(input_shape, classes):
    model = Sequential()

    model.add(Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), padding="valid", activation="relu",
                     kernel_initializer="uniform", input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same",
                     activation="relu", kernel_initializer="uniform"))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same",
                     kernel_initializer="uniform"))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same",
                     kernel_initializer="uniform"))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same",
                     activation="relu", kernel_initializer="uniform"))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=classes, activation="softmax"))

    return model


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
            except Exception as e:
                print(e)
    return np.array(data)


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


if __name__ == '__main__':
    model = convnet(input_shape=(267, 267, 3), classes=1000)

    train = get_training_data("data/chest_xray/chest_xray/train")
    test = get_training_data("data/chest_xray/chest_xray/test")
    val = get_training_data("data/chest_xray/chest_xray/val")
    print(train.shape)

    plot_img(train)

    x_train = []
    y_train = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    x_val = []
    y_val = []

    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)

    x_test = []
    y_test = []

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

    # Normalize the data
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255
    x_test = np.array(x_test) / 255

    x_train = x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)

    x_val = x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)

    x_test = x_test.reshape(-1, img_size, img_size, 1)
    y_test = np.array(y_test)

    # DATA GENERATOR

    data_generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    data_generator.fit(x_train)

    ### fitting the model
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3,
                                                min_lr=0.000001)
    history = model.fit(data_generator.flow(x_train, y_train, batch_size=32), epochs=10,
                        validation_data=data_generator.flow(x_val, y_val), callbacks=[learning_rate_reduction])


