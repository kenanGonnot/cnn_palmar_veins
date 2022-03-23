import gc
import glob
import os

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from tqdm import tqdm
import cv2
import numpy as np

from models.zfnet import zfnet_model


def load_img(path, xdim=180, ydim=180):
    label_names = []
    nmax = 6001
    X = []
    y = []
    count = 0
    #print("Loading images...")
    for dirname in tqdm(os.listdir(path), desc="Loading images...", ncols=100):
        # print("dirname : ", dirname)
        label_names.append(dirname)
        data_path = os.path.join(path + "/" + dirname, '*g')
        # print("data_path: " + data_path)
        files = glob.glob(data_path)
        for f1 in files:
            if count > nmax: break
            # print("files : ", f1)
            #img = cv2.imread(f1)
            #img = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(f1, 0)
            img = cv2.resize(img, (xdim, ydim))
            X.append(np.array(img))
            y.append(dirname)
            count += 1
    print(count, ' images lues')
    X = np.array(X)
    y = np.array(y)
    gc.collect()
    return X, y, label_names


if __name__ == '__main__':
    path = "../data/data_palm_vein/NIR"

    X, y, label_names = load_img(path)


    X = X / 255.
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y)
    print("Preprocessing data")
    print("-----------------------------------------")
    print("\nX shape : {}".format(X.shape))
    print("-----------------------------------------")
    print("y shape : {}\n".format(y.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



    Nombre_classes = len(label_names) - 1
    input_shape = (180, 180, 1)
    model, reduce_lr = zfnet_model(input_shape, Nombre_classes)


