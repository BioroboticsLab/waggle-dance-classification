"""
Tests the dance recognition of the neural network with trained weights by drawing a ROC-Graph. In order to do so it
traverses through the WDD-Folder-Structure and uses every dance for which Ground-Truth-Data exists. The trained network
is used to classify every dance with borders from 0 to 1, in order to test the net for different borders.
Finally the ROC-Graph is plotted.
Usage: testDanceRecognition <validationFolderRoot> <inputModelFile>
:param validationFolderRoot: Root of the Validation Folder, which has dance folders with Ground Truth Data.
:param inputModelFile: Weights that will be used for the neural network.
"""
import numpy as np
import os
import glob
import scipy
import csv
import kerasModel
import sys
import matplotlib.pyplot as plt

from keras.utils import np_utils
from shutil import copytree


def classify_dance(img_array):
    """
    Uses the neural network to give predictions for every single window of a dance.
    :param img_array:
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
    Updates the confusion matrices with the predictions of the windows of one dance, by using every border
     from 0 to 1 in 0.01 steps.
    :param predictions: predictions of the dance
    :param CM: confusion matrices (100,2,2) that will be updated
    :param Y: the actual class of the dance (0 or 1)
    :return: CM: updated confusion matrices
    """
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


def plot_roc_points(CM):
    """
    Plots the confusion matrices in a ROC-curve, one axis for the True Positive Rates and one for
    the False Positive Rates
    :param CM: confusion matrices to be plotted
    """
    fpr_values = []
    tpr_values = []
    for i in range(0, 100, 1):
        tpr = CM[i, 0, 0]/(CM[i, 0, 0] + CM[i, 1, 0])
        fpr = CM[i, 0, 1]/(CM[i, 0, 1] + CM[i, 1, 1])
        fpr_values.append(fpr)
        tpr_values.append(tpr)
    fpr_values = np.asarray(fpr_values)
    tpr_values = np.asarray(tpr_values)
    print(fpr_values)
    print(tpr_values)
    plt.plot(fpr_values, tpr_values)
    plt.axis([0, 1, 0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.grid(True)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.show()


# Get input arguments
validationFolderRoot = sys.argv[1]
inputModelFile = sys.argv[2]

print 'Validation folder root is "', validationFolderRoot
print 'Input model file is "', inputModelFile

# Load the Keras model and the model weight
kM = kerasModel.KerasModel();
model = kM.getModel()
model.load_weights(inputModelFile)

# Init Confusion Matrix
CM = np.zeros((100,2,2))
progress = 0
# Traverse folder structure and build matrix for every single dance so that CNN can test it
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
plot_roc_points(CM);

