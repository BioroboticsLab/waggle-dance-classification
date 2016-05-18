"""
Trains the Keras model of the neural network in kerasModel.py.
Usage: train2DImageGenerator <trainFile> <numberOfEpochs> <outputModelFile> <inputModelFile>
:param trainFile: The .mat - file where Train and Validation Matrix for Training can be found
:param numberOfEpochs: Number of Epochs for the training of the neural network
:param outputModelFile: The name of the file where the weights will be saved. Should end with .h5
:param inputModelFile: Weights will be initialized with the weights from this file. If Errors during initialization occur, there will be a warning and they will be ignored.
"""
import numpy as np
import h5py
import kerasModel
import sys
import os.path

from keras.utils import np_utils, generic_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

trainFile = sys.argv[1]
numberOfEpochs = int(sys.argv[2])
outputModelFile = sys.argv[3]
inputModelFile = sys.argv[4]

print('Input model file is "', inputModelFile)
print('Output model file is "', outputModelFile)
print('Train file is "', trainFile)
print('Number of Epocs is "', numberOfEpochs)

# Load and prepare training data
f = h5py.File(trainFile)
X_train = np.array(f['X_train']).transpose()
Y_train = np.array(f['Y_train'])[0]
X_test = np.array(f['X_test']).transpose()
Y_test = np.array(f['Y_test'])[0]
X_train = X_train.astype("float32")
X_train /= 255
Y_train = np_utils.to_categorical(Y_train, 2)
X_test = X_test.astype("float32")
X_test /= 255
Y_test = np_utils.to_categorical(Y_test, 2)

train_samples = np.size(X_train, 0)

f.close()

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=180,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True)

datagen.fit(X_train)

# Get model of neuronal network and load weights
kM = kerasModel.KerasModel();
model = kM.getModel()

if os.path.exists(inputModelFile):
    try:
        model.load_weights(inputModelFile)
    except: #catch all exceptions
        print('Error during initialization of weights. Ignoring...')
        pass
else:
    print('Input Model File not found. Training without initializing weights.')


# Train model
nb_epoch = numberOfEpochs

model.fit_generator(generator=datagen.flow(X_train, Y_train, batch_size=32), nb_epoch=numberOfEpochs, samples_per_epoch=train_samples, verbose=1, validation_data=(X_test, Y_test), class_weight={0:1, 1:1})
model.save_weights(outputModelFile)
