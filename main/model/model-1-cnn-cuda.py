import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import preprocessing
import PIL
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPooling3D
from tensorflow.keras import utils

# This checks if TF-GPU is installed
# This portion of code was added from the source
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# Hard Coding
# production
# The filepath was changed to be relative to my computer
dir_path_images = '/home/blue/general-assembly/dsir-824/blog/sara-capstone-cuda/data/processed/melspec/'

image_size = (180,180)
batch_size = 64
epochs = 10

# test
# The filepath was changed to be relative to my computer
test_path = '/home/blue/general-assembly/dsir-824/blog/sara-capstone-cuda/data/processed/melspec/baroque/img_150_baroque.png'
image = PIL.Image.open(test_path)
width, height = image.size
print(width,height)

df_train = tf.keras.preprocessing.image_dataset_from_directory(
    dir_path_images,
    image_size=(180,180),
    validation_split=0.2,
    subset='training',
    seed=42,
    batch_size=batch_size,
)

df_val = tf.keras.preprocessing.image_dataset_from_directory(
    dir_path_images,
    image_size=(180,180),
    validation_split=0.2,
    subset='validation',
    seed=42,
    batch_size=batch_size,
)

# I have omitted the matplotlib part to optimize speed for Command Line Execution

# Get Data
df_train = df_train.prefetch(buffer_size=32)
df_val = df_val.prefetch(buffer_size=32)

# Make Keras Model
def make_model(input_shape, num_classes):
    '''
    The function creates a CNN model that can be seen by looking at the 
    model.summary(attribute)
    
    Parts of this model were changed to accept an input that is BGR
    '''
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1. / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    
    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    
    #if num_classes == 2:
    #    activation = "sigmoid"
    #    units = 1
    #else:
    #    activation = "softmax"
    #    units = num_classes
    activation = "softmax"
    units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs), x

# Instantiate Model
model, x = make_model(input_shape=image_size + (3,), num_classes=4)
# The portion showing the network was removed

# Set up model

model.compile(
    #optimizer=keras.optimizers.Adam(1e-3),
    #loss='categorical_crossentropy',
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    # Activate this to run the model as python code, line by line
    # it is slower but it is useful for debugging
    #run_eagerly=True
)

# Changed number of workers to 4 threads
# Enabled Multiprocessing
history = model.fit(df_train, epochs=epochs, validation_data=df_val, workers=8, use_multiprocessing=True)