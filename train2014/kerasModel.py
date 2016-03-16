from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


class KerasModel:
    """Model of a neuronal network"""

    def __init__(self):
        self.data = []

    def getModel(self):
        """
        Returns the model of the neural network.
        :return:
        """
        # Data dimensions
        rows, cols = 20, 20
        image_count = 30

        # Number of convolutional filters to use
        nb_filters = 8
        # Size of pooling area for max pooling
        nb_pool = 2
        # Convolution kernel size
        nb_conv = 3

        # Define model of neural net
        model = Sequential()

        model.add(Convolution2D(16, 5, 5,
                                border_mode='same',
                                input_shape=(image_count, rows, cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(Convolution2D(8,5,5))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(2048))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(7))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        return model;

