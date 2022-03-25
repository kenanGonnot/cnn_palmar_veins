import gc
import base64

import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image_dataset import load_image
from keras_preprocessing.image import img_to_array


def init(architecture_path, weights_path):
    json_file = open(architecture_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # Load the trained model
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    print("------------------------------ Model loaded ------------------------------")
    print(model.summary())
    return model

# def process_image(image):
#     '''
#     Make an image ready-to-use by VGG19
#     '''
#     # convert the image pixels to a numpy array
#     image = img_to_array(image)
#     # reshape data for the model
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     # prepare the image for the VGG model
#     image = preprocess_input(image)
#
#     return image


def model_predict(img_path, model):
    img_shape = 128
    img = load_image(img_path, image_size=(img_shape, img_shape), num_channels=1, interpolation='bilinear')

    # img = image.load_img(img_path, target_size=(img_shape, img_shape), grayscale=True)
    # x = image.img_to_array(img)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # img = cv2.resize(img, ((int(img_shape*1), int(img_shape*1))))
    # x = img.append(np.array(img))
    # x = np.array(x)
    # gc.collect()
    img = img.reshape((1, img_shape, img_shape, 1))
    img = img / 255.
    y_pred = model.predict(img)
    return y_pred


# decode the image coming from the request
def decode_request(req):
    encoded = req["image"]
    decoded = base64.b64decode(encoded)
    return decoded
