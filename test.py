#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:57:51 2022

@author: adugna
"""



# Import, authenticate and initialize the Earth Engine library.
import ee
ee.Initialize()
import helper

import tensorflow as tf
print('Tensorflow version is: ',tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


###########################################
# PARAMETERS
###########################################

#roi
geometry =  ee.Geometry.Polygon(
        [[[103.08000490033993, -2.8225068747308946],
          [103.08000490033993, -2.9521181019620673],
         [103.29217836225399, -2.9521181019620673],
         [103.29217836225399, -2.8225068747308946]]])

geometry2 =     ee.Geometry.Polygon(
        [[[103.28423388261817, -2.666639235594898],
          [103.28423388261817, -2.7983252476718885],
          [103.47786791582129, -2.7983252476718885],
          [103.47786791582129, -2.666639235594898]]])

#Parameters
params = {   # GCS bucket
           'START_DATE': '2021-12-01', 
            'STOP_DATE': '2021-12-31',        
            'ORBIT': 'DESCENDING',
            'RELATIVE_ORBIT_NUMBER':18, 
            'POLARIZATION': 'VVVH',
            'ROI':    geometry,
            'FORMAT': 'DB',
            'CLIP_TO_ROI': True,
            'EXPORT': 'GCS',
            'BUCKET' : 'your-GCS-bucket-name',
            'DRIVE' : '/content/drive',
            'FOLDER' : 'deSpeckNet',
            'USER_ID' : 'users/adugnagirma',
            'IMAGE_PREFIX' : 'deSpeckNet_TEST_PATCH_v3_',
          # Should be the same bands selected during data prep
            'BANDS': ['VV', 'VH'],
            'RESPONSE_TR' : ['VV_median', 'VH_median'],
            'RESPONSE_TU' : ['VV', 'VH'],
            'MASK' : ['VV_mask', 'VH_mask'],
            'KERNEL_SIZE' : 256,
            'KERNEL_SHAPE' : [256, 256],
            'KERNEL_BUFFER' : [128, 128],
            'MODEL_NAME': 'tune'
            }

#process Sentinel 1 image collection
s1_processed = helper.s1_prep(params)
bandNames = s1_processed.first().bandNames().remove('angle')
s1_processed = s1_processed.select(bandNames)
print('Number of images in the collection: ', s1_processed.size().getInfo())

image = s1_processed.first()
# Specify inputs (Sentinel-1 bands) to the model and the response variable.

###########################################
# EXPORT AND INFERENCE
###########################################

# load the saved model
if params['EXPORT'] == 'GCS':
    MODEL_DIR = 'gs://' + params['BUCKET'] + '/' + params['FOLDER'] + '/' + params['MODEL_NAME']
else:
    MODEL_DIR = params['DRIVE'] + '/' + params['FOLDER'] + '/' + params['MODEL_NAME']

#custom_objects={'TransformerBlock': TransformerBlock}
model = tf.keras.models.load_model(MODEL_DIR)
model.summary()

# Run the export. (Run the export only once)
helper.export(image, params)
# Run the prediction.
helper.doPrediction(params)

