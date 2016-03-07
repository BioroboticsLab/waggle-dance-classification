import numpy as np
import os
import glob
import scipy
import csv
import matplotlib.pyplot as plt
import kerasModel

from keras.utils import np_utils
from shutil import copytree


def classify_dance(img_array):
    x_verify = []
    for i in range(img_array.shape[0]-30):
        x_verify.append(img_array[i:i+30,:,:])
    x_verify = np.asarray(x_verify)
    x_verify = x_verify.astype("float32")
    x_verify /= 255
    return model.predict(x_verify, batch_size=128, verbose=0)


def update_confusion_matrix(predictions, CM, dirName):
    predictions_sum = np.sum(predictions, axis=0, dtype="float32")[1]
    mean = predictions_sum/len(predictions)
    border = 0.5
    if mean < border:
	CM[0] += 1
    else:
	CM[1] += 1
	copytree(dirName, '/home/mehmed/Desktop/Clean/' + dirName)
    return CM

kM = kerasModel.KerasModel();
model = kM.getModel()

model.load_weights('my_model_weights_0103w.h5')

#Test model
# Init Confusion Matrix
CM = np.zeros((2))
progress = 0
# traverse folder structure and build matrix for every single dance so that CNN can test it
# Set the directory you want to start from
rootDir = '/home/mehmed/Desktop/20140822'
for dirName, subdirList, fileList in os.walk(rootDir):
   # print('Found directory: %s' % dirName)
    if 'orient.png' in fileList:
            image_list = []
            for fname in glob.glob(dirName + '/image*.png'):
                if (fname != 'orient.png'):
		    im = scipy.misc.imread(fname)[:, :, 1]
                    image_list.append(im)
            image_array = np.asarray(image_list)
            pred = classify_dance(image_array)
            CM = update_confusion_matrix(pred, CM, dirName)
            progress += 1;
            print(progress)
print(CM);

