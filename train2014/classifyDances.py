"""
Classifies the dances of the results of the WDD with the given weights and the neural network in kerasModel.py.
Every positive dance gets copied into the result folder.
Usage: python classify Dances. py <wddOutputRoot> <inputModelFile> <resultFolder>
:param wddOutputRoot: root of the folder with the unfiltered dances.
:param inputModelFile: trained weights of the neural network in kerasModel.py
:param resultFolder: destination of the positively classified dances
"""
import numpy as np
import os
import glob
import scipy
import kerasModel
import sys

from shutil import copytree


def classify_dance(img_array):
    """
    Uses the neural network to give predictions for every single window of a dance.
    :param img_array:
    :return:
    """
    x_verify = []
    for i in range(img_array.shape[0]-30):
        x_verify.append(img_array[i:i+30,:,:])
    x_verify = np.asarray(x_verify)
    x_verify = x_verify.astype("float32")
    x_verify /= 255
    return model.predict(x_verify, batch_size=128, verbose=0)


def update_confusion_matrix(predictions, CM, dirName):
    """
    Updates the confusion matrix with the predictions of the windows of one dance, by using border 0.5.
    Copies positively classified dances into the result folder.
    :param predictions: predictions of the dance
    :param CM: confusion matrix (2,2) that will be updated
    :param dirName: destination of the positively classified dances.
    :return: CM: updated confusion matrix
    """
    predictions_sum = np.sum(predictions, axis=0, dtype="float32")[1]
    mean = predictions_sum/len(predictions)
    border = 0.5
    if mean < border:
        CM[0] += 1
    else:
        CM[1] += 1
        copytree(dirName, resultFolder + dirName)
    return CM

# Get input arguments
wddOutputRoot = sys.argv[1]
inputModelFile = sys.argv[2]
resultFolder = sys.argv[3]

print 'Folder with unfiltered dances is "', wddOutputRoot
print 'Input model file is "', inputModelFile

kM = kerasModel.KerasModel()
model = kM.getModel()

model.load_weights(inputModelFile)

# Init Confusion Matrix
CM = np.zeros((2,2))
progress = 0
# Traverse folder structure and build matrix for every single dance so that CNN can test it
# Set the directory you want to start from
rootDir = wddOutputRoot
for dirName, subdirList, fileList in os.walk(rootDir):
    if 'orient.png' in fileList:
            image_list = []
            for fname in glob.glob(dirName + '/image*.png'):
                if fname != 'orient.png':
                    im = scipy.misc.imread(fname)[:, :, 1]
                    image_list.append(im)
            image_array = np.asarray(image_list)
            pred = classify_dance(image_array)
            CM = update_confusion_matrix(pred, CM, dirName)
            progress += 1
            print(progress)
print(CM)

