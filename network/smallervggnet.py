from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        #__________Block 1__________#
        # Conv => RELU => Pool
        #32 filters, 3x3 kernel
        model.add(Conv2D(32, (3,3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        #3x3 pool size
        model.add(MaxPooling2D(pool_size=(3,3)))
        #Drop 25% of nodes from network randomly to increase
        #model redundancy
        model.add(Dropout(0.25))

        #__________Block 2__________#
        #(Conv => RELU) * 2 => Pool
        #increased filter size from 32 to 64
        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        #decreased pooling size
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        #__________Block 3__________#
        #(Conv => RELU) * 2 => Pool
        #increased filter size from 32 to 64
        model.add(Conv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3,3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        #maintain pooling size
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        #__________Block 4__________#
        #FC => RELU
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        #Softmax Classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        #return the model
        return model

