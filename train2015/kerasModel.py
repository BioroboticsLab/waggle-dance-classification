from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers

class KerasModel:
    """Model of a neuronal network"""

    def __init__(self):
        self.data = []
        self.rows = 30
        self.cols = 30
        self.image_count = 20

    def getModel(self):
        """
        Returns the model of the neural network.
        :return:
        """
        # Data dimensions
        rows, cols = self.rows, self.cols
        image_count = self.image_count

        # Define model of neural net
        model = Sequential()

        model.add(Convolution2D(32, 2, 2,
                                border_mode='same',
                                input_shape=(image_count, rows, cols)))
        model.add(Activation('relu'))
    #    model.add(MaxPooling2D(pool_size=(2, 2)))
    #    model.add(Dropout(0.2))
        model.add(Convolution2D(16, 2, 2))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(8,2,2))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(2048))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(7))
        model.add(Activation('relu'))
        model.add(Dense(2)) 
        model.add(Activation('softmax'))
        adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=["accuracy"])
        return model;

    def get_image_count(self):
        return self.image_count;
