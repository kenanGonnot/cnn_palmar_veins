from keras import Sequential
from keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


def depthwise_conv_model():
    model = Sequential()
    model.add(SeparableConv2D(64, kernel_size=3, depthwise_initializer='he_uniform', activation='LeakyReLU',
                              input_shape=(224, 224, 1), padding='same'))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(SeparableConv2D(128, kernel_size=3, activation='LeakyReLU', padding='same',
                              depthwise_initializer='he_uniform', ))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(SeparableConv2D(256, kernel_size=3, activation='LeakyReLU', padding='same',
                              depthwise_initializer='he_uniform', ))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(SeparableConv2D(512, kernel_size=3, activation='LeakyReLU', padding='same',
                              depthwise_initializer='he_uniform', ))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='LeakyReLU'))
    model.add(Dense(4096, activation='LeakyReLU'))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())
    print("\n ================= depthwise CONV model ================= \n")

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model = depthwise_conv_model()

    model.compile(optimizer=Adam(.001), loss='binary_crossentropy', metrics=['accuracy'])
