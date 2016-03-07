import scipy.io as sio
import numpy as np
import h5py
import kerasModel

from keras.utils import np_utils, generic_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Load and prepare training data
f = h5py.File('gtalltrain.mat')
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

model.load_weights('my_model_weights_0103v2.h5')

# Train model
nb_epoch = 3

for e in range(nb_epoch):
    print('Epoch', e)
    progbar = generic_utils.Progbar(X_train.shape[0])
    # batch train with real time data augmentation
    for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=128, shuffle=True):
        loss, accuracy = model.train_on_batch(X_batch, Y_batch, accuracy=True, class_weight={0:1, 1:5})
        progbar.add(X_batch.shape[0], values=[("loss", loss), ("acc", accuracy)])
    progbarTest = generic_utils.Progbar(X_test.shape[0])
    loss, accuracy = model.evaluate(X_test, Y_test, batch_size=128, show_accuracy=True, verbose=0)
    progbarTest.add(X_test.shape[0], values=[("val_loss", loss), ("val_acc", accuracy)])

#model.fit(X_train, Y_train, batch_size=128, class_weight={0:1, 1:5}, nb_epoch=42, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))

model.save_weights('my_model_weights_0103w.h5')
