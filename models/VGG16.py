
from keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout
from keras.models import Sequential
from mlxtend.evaluate import accuracy
from tensorflow.keras.optimizers import SGD


def vgg_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', activation='LeakyReLU', input_shape=(224, 224, 1),
                     padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_uniform', activation='LeakyReLU',
                     padding='same'))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(
        Conv2D(128, kernel_size=(3, 3), activation='LeakyReLU', padding='same', kernel_initializer='he_uniform', ))
    model.add(
        Conv2D(128, kernel_size=(3, 3), activation='LeakyReLU', padding='same', kernel_initializer='he_uniform', ))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(
        Conv2D(256, kernel_size=(3, 3), activation='LeakyReLU', padding='same', kernel_initializer='he_uniform', ))
    model.add(
        Conv2D(256, kernel_size=(3, 3), activation='LeakyReLU', padding='same', kernel_initializer='he_uniform', ))
    model.add(
        Conv2D(256, kernel_size=(3, 3), activation='LeakyReLU', padding='same', kernel_initializer='he_uniform', ))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(
        Conv2D(512, kernel_size=(3, 3), activation='LeakyReLU', padding='same', kernel_initializer='he_uniform', ))
    model.add(
        Conv2D(512, kernel_size=(3, 3), activation='LeakyReLU', padding='same', kernel_initializer='he_uniform', ))
    model.add(
        Conv2D(512, kernel_size=(3, 3), activation='LeakyReLU', padding='same', kernel_initializer='he_uniform', ))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='LeakyReLU'))
    model.add(Dense(4096, activation='LeakyReLU'))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())
    model.compile(optimizer=SGD(.001), loss='binary_crossentropy', metrics='accuracy')

    return model


if __name__ == '__main__':
    model = vgg_model()

    model.compile(optimizer=SGD(.001), loss='binary_crossentropy', metrics=['accuracy'])
