import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications import VGG16

class BuildModel:

    def __init__(self):
        self.model = 0
        self.hist = 0

    def GetModel(self, input_shape, no_of_class=3, learning_rate=1e-3):
        #Vgg model - Pre Trained model on Imagenet Dataset
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

        #Fully connected Layers
        av = GlobalAveragePooling2D()(vgg.output)
        fl = Flatten()(av)
        drop1 = Dropout(0.5)(fl)
        d1 = Dense(256, activation='relu')(drop1)
        drop2 = Dropout(0.5)(d1)
        output = Dense(no_of_class, activation='softmax')(drop2)

        self.model = Model(vgg.input, output)
        for i in range(17): 
            self.model.layers[i].trainable = False

        adam = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    def ModelLayerInfo(self):
        return self.model.summary()