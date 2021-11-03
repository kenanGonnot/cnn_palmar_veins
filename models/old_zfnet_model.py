from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf

def old_zfnet_model(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), padding="valid", activation="LeakyReLU",
                     kernel_initializer="uniform", input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same",
                     activation="LeakyReLU", kernel_initializer="uniform"))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
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
    model.compile(optimizer='nadam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics='accuracy')

    return model


if __name__ == '__main__':
    old_zfnet_model()