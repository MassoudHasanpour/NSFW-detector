# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:54:30 2021

@author: Masoud H
"""

import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import SGD


val_acc = 0
train_dir = '/home/asheghabadi/pic/training'
validation_dir = '/home/asheghabadi/pic/testing/testing'
validation_dir = '/home/asheghabadi/pic/testing/testing'

                                   # shear_range=0.45,


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=60,
                                   width_shift_range=0.45,
                                   height_shift_range=0.45,
                                   brightness_range = [0.3 , 1.0],
                                   zoom_range=0.6,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')  # with data augmentation for train set

valid_datagen = ImageDataGenerator(rescale=1./255)  # no augmentation for validation set

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=10,
                                                    class_mode='categorical',
                                                    target_size=(299, 299))



validation_generator = valid_datagen.flow_from_directory(validation_dir,
                                                         batch_size=10,
                                                         class_mode='categorical',
                                                         target_size=(299, 299))


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') is not None and logs.get('val_acc') is not None and (logs.get('acc') > 0.99 and logs.get('val_acc') > 0.99):
            print("\nCancelling training as model has reached 99% accuracy and 99% validation accuracy!")
            self.model.stop_training = True
           
            
def plot_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()
    
    
img_width, img_height = 150, 150

# Import Inception v3 Model
pre_trained_model = InceptionV3(input_shape=(299, 299, 3), include_top=False, weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False
    
pre_trained_model.summary()
plot_model(pre_trained_model, to_file='inception_v3_model.png', show_shapes=False, show_layer_names=True)

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.25)(x)
# x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(2, activation='softmax')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(lr=0.00005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.compile(optimizer=SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

model.summary()
plot_model(model, to_file='inception_v3_with_dense_layers_model.png', show_shapes=False, show_layer_names=True)

logs={}

cp_Callbacks = tf.keras.callbacks.ModelCheckpoint('/home/asheghabadi/pic/model_3/', monitor = 'val_accuracy', 
                                   verbose = 2, save_best_only = True, mode = 'max') 
# callbacks = myCallback()
history = model.fit_generator(generator=train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=30,
                              epochs=200,
                              validation_steps=30,
                              verbose=2,
                              callbacks = [cp_Callbacks])

val_acc = history.history['val_accuracy']
plot_result(history)

model.save('my_model3')

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")