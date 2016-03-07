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


def update_confusion_matrix(predictions, CM, Y):
    predictions_sum = np.sum(predictions, axis=0, dtype="float32")[1]
    mean = predictions_sum/len(predictions)
    for i in range(0, 100, 1):
        border = i*0.01
        if mean < border:
            if Y == 0:
                CM[i, 1, 1] += 1
            elif Y == 1:
                CM[i, 1, 0] += 1
        else:
            if Y == 0:
                CM[i, 0, 1] += 1
            elif Y == 1:
                CM[i, 0, 0] += 1
    return CM


def save_roc_points(CM):
    fpr_values = []
    tpr_values = []
    for i in range(0, 100, 1):
        tpr = CM[i, 0, 0]/(CM[i, 0, 0] + CM[i, 1, 0])
        fpr = CM[i, 0, 1]/(CM[i, 0, 1] + CM[i, 1, 1])
        fpr_values.append(fpr)
        tpr_values.append(tpr)
    fpr_values = np.asarray(fpr_values)
    tpr_values = np.asarray(tpr_values)
    np.save('fpr_values_vaw', fpr_values)
    np.save('tpr_values_vaw', tpr_values)


kM = kerasModel.KerasModel();
model = kM.getModel()

model.load_weights('my_model_weights_0103w.h5')

#Test model
# Init Confusion Matrix
CM = np.zeros((100,2,2))
progress = 0
# traverse folder structure and build matrix for every single dance so that CNN can test it
# Set the directory you want to start from
rootDir = '/home/mehmed/Desktop/GTV2014'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    if 'gt.csv' in fileList:
        with open(dirName+'/gt.csv', 'rb') as csvfile:
            spamReader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            Y = 0
            for row in spamReader:
                Y = row[0]
            if Y == 'j':
                Y = 1
            elif Y == 'n':
                Y = 0
            else:
                Y = -1
            image_list = []
            for fname in glob.glob(dirName + '/image*.png'):
                im = scipy.misc.imread(fname)[:, :, 1]
                image_list.append(im)
            image_array = np.asarray(image_list)
            pred = classify_dance(image_array)
            CM = update_confusion_matrix(pred, CM, Y)
            progress += 1;
            print(progress)
save_roc_points(CM);

