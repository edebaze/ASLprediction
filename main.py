import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Dropout,BatchNormalization,Conv2D,MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import layers
import pickle


if __name__ == '__main__':
    training_dir = 'data'
    labels = sorted(os.listdir(training_dir))

    # =====================================================================
    # DATA GENERATORS
    data_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        brightness_range=[0.8, 1.0],
        zoom_range=[1.0, 1.2],
        validation_split=0.1
    )

    train_generator = data_generator.flow_from_directory(
        training_dir,
        target_size=(200, 200),
        shuffle=True,
        seed=13,
        class_mode='categorical',
        batch_size=64,
        subset="training"
    )

    validation_generator = data_generator.flow_from_directory(
        training_dir,
        target_size=(200, 200),
        shuffle=True,
        seed=13,
        class_mode='categorical',
        batch_size=64,
        subset="validation"
    )

    # =====================================================================
    # CREATE MODEL

    inception_v3_model = keras.applications.inception_v3.InceptionV3(
        input_shape=(200, 200, 3),
        include_top=False,
        weights='imagenet'
    )

    inception_v3_model.summary()

    inception_output_layer = inception_v3_model.get_layer('mixed7')
    print('Inception model output shape:', inception_output_layer.output_shape)

    inception_output = inception_v3_model.output

    layers = layers.GlobalAveragePooling2D()(inception_output)
    layers = layers.Dense(1024, activation='relu')(layers)
    layers = layers.Dense(51, activation='softmax')(layers)

    model = Model(inception_v3_model.input, x)

    model.compile(
        optimizer=SGD(lr=1e-4, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['acc']
    )
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # =====================================================================
    # CREATE CALLBACK
    LOSS_THRESHOLD = 0.2
    ACCURACY_THRESHOLD = 0.95

    class ModelCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('val_loss') <= LOSS_THRESHOLD and logs.get('val_acc') >= ACCURACY_THRESHOLD:
                print("\nReached", ACCURACY_THRESHOLD * 100, "accuracy, Stopping!")
                self.model.stop_training = True


    callback = ModelCallback()

    # =====================================================================
    # TRAIN
    history = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=200,
        validation_steps=50,
        epochs=50,
        callbacks=[callback]
    )
    model.save('transferlearning.h5')

    # =====================================================================
    # EVALUATE METRICS
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()
