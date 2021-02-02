import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.preprocessing import image

def TrainGenerator():
    train_idg = image.ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True
        )

    train_generator = train_idg.flow_from_directory(
        './Dataset/CT_Images/Train/',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical'
        )

    return train_generator

def ValGenerator():
    val_idg = image.ImageDataGenerator(
        rescale=1./255
        )
        
    val_generator = val_idg.flow_from_directory(
        './Dataset/CT_Images/test',
        target_size=(224,224),
        batch_size=24,
        class_mode='categorical'
        )

    return val_generator