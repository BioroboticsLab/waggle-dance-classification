from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


class KerasModel:
    """Model of a neuronal network"""

    def __init__(self):
        self.data = []
        self.rows = 20
        self.cols = 20
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

        model.add(Convolution2D(16, 5, 5,
                                border_mode='same',
                                input_shape=(image_count, rows, cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(8, 3, 3))
        model.add(Activation('relu'))
        model.add(Convolution2D(8,5,5))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
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

    def get_image_count(self):
        return self.image_count;
