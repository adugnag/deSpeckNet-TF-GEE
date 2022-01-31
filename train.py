#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:02:21 2022

@author: adugna
"""

import tensorflow as tf
print('Tensorflow version is: ',tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#tf.enable_eager_execution()
import helper
import os

###########################################
# PARAMETERS
###########################################

params = {   # GCS bucket
            'EXPORT': 'GCS',
            'BUCKET' : 'GCS-bucket-name',
            'DRIVE' : '/content/drive',
            'FOLDER' : 'deSpeckNet',
            'TRAINING_BASE' : 'training_deSpeckNet_DUAL_Median_mask_test',
            'EVAL_BASE' : 'eval_deSpeckNet_DUAL_median_mask_test',
            'MODE' : 'training',
          # Should be the same bands selected during data prep
            'BANDS': ['VV', 'VH'],
            'RESPONSE_TR' : ['VV_median', 'VH_median'],
            'RESPONSE_TU' : ['VV', 'VH'],
            'MASK' : ['VV_mask', 'VH_mask'],
            'KERNEL_SIZE' : 40,
            'KERNEL_SHAPE' : [40, 40],
            'KERNEL_BUFFER' : [20, 20],
          # Specify model training parameters.
            'BATCH_SIZE' : 16,
            'TRAIN_SIZE':32000,
            'EVAL_SIZE':8000,
            'EPOCHS' : 50,
            'BUFFER_SIZE': 2000,
            'TV_LOSS' : False,
            'DEPTH' : 17,
            'FILTERS' : 64,
            'MODEL_NAME': 'deSpeckNet_v0'
            }


if params['MODE'] == 'training':
  FEATURES = params['BANDS'] + params['RESPONSE_TR'] + params['MASK']
else:
  FEATURES = params['BANDS']  + params['RESPONSE_TU']

# Specify the feature columns and create a feature dictionary for the data pipeline.
COLUMNS = [tf.io.FixedLenFeature(shape=params['KERNEL_SHAPE'], dtype=tf.float32) for k in FEATURES]
FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

IMAGE_CHANNELS = len(params['BANDS'])

if params['EXPORT'] == 'GCS':
    MODEL_DIR = 'gs://' + params['BUCKET'] + '/' + params['FOLDER'] + '/' + params['MODEL_NAME']
else:
    MODEL_DIR = params['DRIVE'] + '/' + params['FOLDER'] + '/' + params['MODEL_NAME']

###########################################
# TRAINING DATA
###########################################

#Use the tf.data api to build our data pipeline
training = helper.get_training_dataset(params, FEATURES, FEATURES_DICT)
evaluation = helper.get_eval_dataset(params)

#Check data
print(iter(training.take(1)).next())

###########################################
# BUILD MODEL
###########################################

model = helper.deSpeckNet(depth=params['DEPTH'],filters=params['FILTERS'],image_channels=IMAGE_CHANNELS)
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

#For fine tuning
if params['MODE'] != 'training':
    model = tf.keras.models.load_model(MODEL_DIR)

###########################################
# TRAIN MODEL
###########################################

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension.
# %load_ext tensorboard

import tensorboard
print(tensorboard.__version__)

# Define the TensorBoard callback.
os.mkdir(params['MODEL_NAME'])
logdir= params['MODEL_NAME']
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(helper.lr_schedule)

if params['TV_LOSS']:
  loss_funcs = {'clean1': 'mean_squared_error','clean1':helper.TVloss,'noisy1' : 'mean_squared_error'}
  loss_weights = {'clean1': 100.0, 'clean1':0.0, 'noisy1': 1.0}
else:
    loss_funcs = {'clean1': 'mean_squared_error','noisy1' : 'mean_squared_error'}
    loss_weights = {'clean1': 100.0,'noisy1': 1.0}

#Compile
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_funcs, loss_weights=loss_weights)

model.fit(
    x=training, 
    epochs=params['EPOCHS'],
    steps_per_epoch=int(params['TRAIN_SIZE'] / params['BATCH_SIZE']), 
    validation_data=evaluation,
    validation_steps=int(params['EVAL_SIZE'] / params['BATCH_SIZE']),
    callbacks=[tensorboard_callback, lr_scheduler])

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir params['MODEL_NAME']

# Save the trained model
if params['MODE'] == 'training':
# Save the trained model
  model.save(MODEL_DIR, save_format='tf')
else:
  MODEL_DIR = MODEL_DIR + '_' + 'tune'
  model.save(MODEL_DIR, save_format='tf')
