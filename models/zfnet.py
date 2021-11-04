import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Lambda, Dropout
from keras.models import Sequential
from mlxtend.evaluate import accuracy
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.metrics import TopKCategoricalAccuracy


def zfnet_model(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), padding="valid", activation="LeakyReLU",
                     kernel_initializer="uniform", input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Lambda(lambda x: tf.image.per_image_standardization(x)))

    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same",
                     activation="LeakyReLU", kernel_initializer="uniform"))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # model.add(Lambda(lambda x: tf.image.per_image_standardization(x)))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same",
                     kernel_initializer="uniform"))

    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same",
                     kernel_initializer="uniform"))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same",
                     activation="LeakyReLU", kernel_initializer="uniform"))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation="LeakyReLU"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="LeakyReLU"))
    model.add(Dropout(0.5))
    model.add(Dense(units=classes, activation="softmax"))

    print(model.summary())
    print("\n ================= ZFNET model ================= \n")
    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy',
    #               metrics=['accuracy', TopKCategoricalAccuracy(1)])
    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.1, patience=1, min_lr=0.00001)
    # reduce_lr=0
    return model, reduce_lr


if __name__ == '__main__':
    input_shape = (224, 224, 1)
    model = zfnet_model(input_shape, 2)
    # train, val = load_img("../data/chest_xray/chest_xray/train", "data/chest_xray/chest_xray/val")

    # plot_img(train)
    # random.shuffle(train)

    # x_train = []
    y_train = []
    #
    for feature, label in train:
        # x_train.append(feature)
        y_train.append(label)
    #
    # # print(x_train)
    # print(y_train)
    #
    # x_val = []
    # y_val = []
    #
    # for feature, label in val:
    #     x_val.append(feature)
    #     y_val.append(label)
    #
    # x_test = []
    # y_test = []
    #
    # for feature, label in test:
    #     x_test.append(feature)
    #     y_test.append(label)
    #
    # # Normalize the data
    # x_train = np.array(x_train) / 255
    # x_val = np.array(x_val) / 255
    # x_test = np.array(x_test) / 255
    # #
    # x_train = x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)
    #
    # x_val = x_val.reshape(-1, img_size, img_size, 1)
    # y_val = np.array(y_val)
    #
    # x_test = x_test.reshape(-1, img_size, img_size, 1)
    # y_test = np.array(y_test)
    #
    # DATA GENERATOR

    # data_generator = ImageDataGenerator(
    #     featurewise_center=False,  # set input mean to 0 over the dataset
    #     samplewise_center=False,  # set each sample mean to 0
    #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #     samplewise_std_normalization=False,  # divide each input by its std
    #     zca_whitening=False,  # apply ZCA whitening
    #     rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    #     zoom_range=0.1,  # Randomly zoom image
    #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #     horizontal_flip=True,  # randomly flip images
    #     vertical_flip=False)  # randomly flip images
    #
    # data_generator.fit(x_train)

    model = zfnet_model(input_shape=(267, 267, 3), classes=1000)

    ### fitting the models
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3,
                                                min_lr=0.000001)
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    # history = models.fit(train, y_train, batch_size=10,epochs=10, callbacks=[learning_rate_reduction])
    model.fit(
        train,
        validation_data=val,
        epochs=3
    )

    # history = models.fit(x_train, y_train, batch_size=10,epochs=10,
    #                         validation_data=data_generator.flow(x_val, y_val), callbacks=[learning_rate_reduction])

    # accuracy = models.evaluate(x_test, y_test, verbose=0)

    print(accuracy)
