import scipy.io as sio
import numpy as np
import h5py

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# Data dimensions
rows, cols = 30, 30
imagecount = 30
f = h5py.File('mixedMat.mat')
X_train = np.array(f['X_train']).transpose()
Y_train = np.array(f['Y_train'])[0]
X_test = np.array(f['X_test']).transpose()
Y_test = np.array(f['Y_test'])[0]
sio.savemat('singleback.mat', {'X_train':X_train} )
X_train = X_train.astype("float32")
X_train /= 255
Y_train = np_utils.to_categorical(Y_train, 2)
X_test = X_test.astype("float32")
X_test /= 255
Y_test = np_utils.to_categorical(Y_test, 2)

f.close()

# number of convolutional filters to use
nb_filters = 8
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#Define model of neural net
model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='full',
                        input_shape=(imagecount, rows, cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

#Test model
model.fit(X_train, Y_train, batch_size=128, nb_epoch=42, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))

model.save_weights('my_model_weights.h5')
