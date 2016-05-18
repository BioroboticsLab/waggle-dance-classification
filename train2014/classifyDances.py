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
from helperFunctions import classify_dance


def update_confusion_matrix(predictions, CM, dirName, resultFolder):
    """
    Updates the confusion matrix with the predictions of the windows of one dance, by using a fixed value for the border.
    Copies positively classified dances into the result folder.
    :param predictions: predictions of the dance
    :param CM: confusion matrix (2,2) that will be updated
    :param dirName: destination of the positively classified dances.
    :return: CM: updated confusion matrix
    """
    predictions_sum = np.sum(predictions, axis=0, dtype="float32")[1]
    mean = predictions_sum/len(predictions)
    border = 0.35
    if mean < border:
        CM[0] += 1
    else:
        CM[1] += 1
        copytree(dirName, resultFolder + dirName)
    return CM

def main():
    # Get input arguments
    wddOutputRoot = sys.argv[1]
    inputModelFile = sys.argv[2]
    resultFolder = sys.argv[3]
    
    print('Folder with unfiltered dances is "', wddOutputRoot)
    print('Input model file is "', inputModelFile)
    
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
        print(dirName)
        if 'orient.png' in fileList:
                image_list = []
                for fname in glob.glob(dirName + '/image*.png'):
                    if fname != 'orient.png':
                        im = scipy.misc.imread(fname)[:, :, 1]
                        image_list.append(im)
                image_array = np.asarray(image_list)
                pred = classify_dance(image_array, model, kM.get_image_count())
                CM = update_confusion_matrix(pred, CM, dirName, resultFolder)
                progress += 1
                print(progress)
    print(CM)

if __name__=="__main__":
    main()

