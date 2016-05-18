"""
Given a folder with Ground Truth Data, this script checks if the neural network classifies every dance correctly.
If not the dance is copied to a folder in given location. There will be two folders, one for false positives and one
for false negatives.
Usage: python testDanceRecognitionExtractWrong.py <validationFolder> <inputModelFile> <outputFolder>
:param validationFolder: The root of the folder with the Ground Truth data that should be used for validation
:param inputModelFile: The weights that should be tested for the neural network
:param outputFolder: The folder in which the wrong classifications will be saved.
"""
import numpy as np
import os
import glob
import scipy
import csv
import kerasModel
import sys

from shutil import copytree
from helperFunctions import classify_dance


def update_confusion_matrix(predictions, CM, Y, dirName, outputFolder):
    """
    Updates the confusion matrix with the predictions of the windows of one dance, by using a border of 0.5.
    Additionally wrong classifications will be copied into the folder validationFolder.
    :param predictions: predictions of the dance
    :param CM: confusion matrix (2,2) that will be updated
    :param Y: the actual class of the dance (0 or 1)
    :return: CM: updated confusion matrix
    """
    predictions_sum = np.sum(predictions, axis=0, dtype="float32")[1]
    mean = predictions_sum/len(predictions)
    border = 0.66
    if mean < border:
        if Y == 0:
            CM[1, 1] += 1
        elif Y == 1:
            CM[1, 0] += 1
            copytree(dirName, outputFolder + '/BL_Dance' + dirName)
    else:
        if Y == 0:
            CM[0, 1] += 1
            copytree(dirName, outputFolder + '/TR_Dance' + dirName)
        elif Y == 1:
            CM[0, 0] += 1
    return CM

def main():
    validationFolder = sys.argv[1]
    inputModelFile = sys.argv[2]
    outputFolder = sys.argv[3]
    
    print('Validation folder is "', validationFolder)
    print('Input model file is "', inputModelFile)
    print('Output folder is "', outputFolder)
    
    kM = kerasModel.KerasModel();
    model = kM.getModel()
    
    model.load_weights(inputModelFile)
    
    # Init Confusion Matrix
    CM = np.zeros((2,2))
    progress = 0
    # traverse folder structure and build matrix for every single dance so that CNN can test it
    # Set the directory you want to start from
    rootDir = validationFolder
    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        if 'gt.csv' in fileList:
            with open(dirName+'/gt.csv', 'rt') as csvfile:
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
                    continue
                image_list = []
                for fname in glob.glob(dirName + '/image*.png'):
                    im = scipy.misc.imread(fname)[:, :, 1]
                    image_list.append(im)
                image_array = np.asarray(image_list)
                pred = classify_dance(image_array, model, kM.get_image_count())
                CM = update_confusion_matrix(pred, CM, Y, dirName, outputFolder)
                progress += 1;
                print(progress)
    print(CM);

if __name__=="__main__":
    main()

