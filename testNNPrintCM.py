"""
Traverses the WDD-Folder-Structure and for every dance with Ground-Truth-Data it updates the CM with the results of every window within the dance.
The root of the WDD-Folder-Structure is given static inside the program code.
Usage: testNNPrintCM <validationFolderRoot> <inputModelFile>
:param validationFolderRoot: Root of the Validation Folder, which has dance folders with Ground Truth Data.
:param inputModelFile: Weights that will be used for the neural network.
"""
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
    """
    Uses the neural network to give predictions for every single window of a dance.
    :param img_array
    """
    x_verify = []
    for i in range(img_array.shape[0]-30):
        x_verify.append(img_array[i:i+30,:,:])
    x_verify = np.asarray(x_verify)
    x_verify = x_verify.astype("float32")
    x_verify /= 255
    return model.predict(x_verify, batch_size=128, verbose=0)


def update_confusion_matrix(predictions, CM, Y):
    """
    Updates the confusion matrices with the predictions of the windows of one dance, by using a border of 0.5
    :param predictions: predictions of the dance
    :param CM: confusion matrix (2,2) that will be updated
    :param Y: the actual class of the dance (0 or 1)
    :return: updated confusion matrix
    """
    print(predictions)
    for i in predictions[:,1]:
        border = 0.50
        mean = i
        if mean < border:
            if Y == 0:
                CM[1, 1] += 1
            elif Y == 1:
                CM[1, 0] += 1
        else:
            if Y == 0:
                CM[0, 1] += 1
            elif Y == 1:
                CM[0, 0] += 1
    return CM

# Get input arguments
validationFolderRoot = sys.argv[1]
inputModelFile = sys.argv[2]

print 'Validation folder root is "', validationFolderRoot
print 'Input model file is "', inputModelFile

kM = kerasModel.KerasModel()
model = kM.getModel()

model.load_weights(inputModelFile)

# Init Confusion Matrix
CM = np.zeros((2,2))
progress = 0
# traverse folder structure and build matrix for every single dance so that CNN can test it
# Set the directory you want to start from
rootDir = validationFolderRoot
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
    print(CM)
