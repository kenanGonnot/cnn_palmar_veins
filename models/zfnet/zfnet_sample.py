import tensorflow as tf

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images[:1000]
training_labels = training_labels[:1000]
test_images = test_images[:100]
test_labels = test_labels[:100]

training_images = tf.map_fn(lambda i: tf.stack([i] * 3, axis=-1), training_images).numpy()
test_images = tf.map_fn(lambda i: tf.stack([i] * 3, axis=-1), test_images).numpy()

training_images = tf.image.resize(training_images, [224, 224]).numpy()
test_images = tf.image.resize(test_images, [224, 224]).numpy()

training_images = training_images.reshape(1000, 224, 224, 3)
training_images = training_images / 255.0
test_images = test_images.reshape(100, 224, 224, 3)
test_images = test_images / 255.0

training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

num_len_train = int(0.8 * len(training_images))

ttraining_images = training_images[:num_len_train]
ttraining_labels = training_labels[:num_len_train]

valid_images = training_images[num_len_train:]
valid_labels = training_labels[num_len_train:]

training_images = ttraining_images
training_labels = ttraining_labels

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(96, (7, 7), strides=(2, 2), activation='relu',
                           input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(3, strides=2),
    tf.keras.layers.Lambda(lambda x: tf.image.per_image_standardization(x)),

    tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(3, strides=2),
    tf.keras.layers.Lambda(lambda x: tf.image.per_image_standardization(x)),

    tf.keras.layers.Conv2D(384, (3, 3), activation='relu'),

    tf.keras.layers.Conv2D(384, (3, 3), activation='relu'),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(3, strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(4096),

    tf.keras.layers.Dense(4096),

    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9), \
              loss='categorical_crossentropy', \
              metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(5)])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', \
                                                 factor=0.1, patience=1, \
                                                 min_lr=0.00001)

model.fit(training_images, training_labels, batch_size=128, \
          validation_data=(valid_images, valid_labels), \
          epochs=90, callbacks=[reduce_lr])