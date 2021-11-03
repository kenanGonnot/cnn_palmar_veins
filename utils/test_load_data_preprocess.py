import os
import warnings

from model.VGG16 import vgg_model
from model.depthwise_conv import depthwise_conv_model
from model.zfnet import zfnet_model
from utils import load_augmented_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore tensorflow warnings
warnings.filterwarnings("ignore")

batch_size = 32
img_height = 224
img_width = 224
input_shape = (img_height, img_width, 1)


if __name__ == '__main__':
    test_dir = "../data/chest_xray/chest_xray/test"
    train_dir = "../data/chest_xray/chest_xray/train"
    # val_dir = "../data/chest_xray/chest_xray/val"

    train, valid = load_augmented_data(train_dir, test_dir)

    # model = zfnet_model(input_shape, 1)
    # model = vgg_model()
    model = depthwise_conv_model()

    print("Training...")
    model.fit(train, validation_data=valid, epochs=5, batch_size=5)

    val = model.evaluate(valid, steps=200)
    print("Accuracy: " + str(val)[1])
