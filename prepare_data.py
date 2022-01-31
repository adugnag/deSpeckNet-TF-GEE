#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:02:48 2022

@author: adugna
"""


# Import, and initialize the Earth Engine library. (Earth engine should be authenticated on the local machine )
import ee
ee.Initialize()
import helper

###########################################
# PARAMETERS
###########################################

#ROI
geometry = ee.Geometry.Polygon(
        [[[103.08000490033993, -2.8225068747308946],
          [103.08000490033993, -2.9521181019620673],
         [103.29217836225399, -2.9521181019620673],
         [103.29217836225399, -2.8225068747308946]]])

#parameters
params = {  'START_DATE': '2021-01-01', 
            'STOP_DATE': '2021-12-31',        
            'ORBIT': 'DESCENDING',
            'RELATIVE_ORBIT_NUMBER':18, 
            'POLARIZATION': 'VVVH',
            'ROI':    geometry,
            'FORMAT': 'DB',
            'CLIP_TO_ROI': False,
          # GCS bucket
            'EXPORT': 'GCS',
            'BUCKET' : 'senalerts_dl3',
            'DRIVE' : '/content/drive',
            'FOLDER' : 'deSpeckNet',
            'TRAINING_BASE' : 'training_deSpeckNet_DUAL_Median_mask_test',
            'EVAL_BASE' : 'eval_deSpeckNet_DUAL_median_mask_test',
            'MODE' : 'training',
            'KERNEL_SIZE' : 40,
            'KERNEL_SHAPE' : [40, 40],
            'KERNEL_BUFFER' : [20, 20]
            }

#process Sentinel 1 image collection
s1_processed = helper.s1_prep(params)
bandNames = s1_processed.first().bandNames().remove('angle')
s1_processed = s1_processed.select(bandNames)
print('Number of images in the collection: ', s1_processed.size().getInfo())

#Uncomment for larger scenes
#image = s1_processed.mosaic()
image = s1_processed.first()
bandNames = image.bandNames().getInfo()
label =s1_processed.reduce(ee.Reducer.median())
stddev = s1_processed.reduce(ee.Reducer.stdDev())
#Mask out pixels with high stdDev. Threshold is higher as the data is in dB.
maskBand = ['VV_mask', 'VH_mask']
#maskBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mask'))
mask = stddev.lte(2.0).rename(maskBand)

###########################################
# FEATURES
###########################################

# Specify inputs (Sentinel-1 bands) to the model and the response variable.
BANDS = image.bandNames().getInfo()
RESPONSE_TR = label.bandNames().getInfo()
RESPONSE_TU = image.bandNames().getInfo()
MASK = mask.bandNames().getInfo()

if params['MODE'] == 'training':
  FEATURES = BANDS + RESPONSE_TR + MASK
else:
  FEATURES = BANDS + RESPONSE_TU

print('List of feature names in input: ', FEATURES)

###########################################
# SAMPLING
###########################################

#Sampling polygons. If using new areas please adjust these polygons

train1 = ee.Geometry.Polygon(
        [[[103.23827668989107, -2.82473762348129],
          [103.23827668989107, -2.8730863008895193],
          [103.28908845746919, -2.8730863008895193],
          [103.28908845746919, -2.82473762348129]]])
          
train2 = ee.Geometry.Polygon(
        [[[103.08275148237153, -2.826109245073874],
          [103.08275148237153, -2.891945160858568],
          [103.1740753349106, -2.891945160858568],
          [103.1740753349106, -2.826109245073874]]])
          
train3 = ee.Geometry.Polygon(
        [[[103.12017366254732, -2.887830527175443],
          [103.12017366254732, -2.94954845545005],
          [103.245829790477, -2.94954845545005],
          [103.245829790477, -2.887830527175443]]])

train4 = geometry = ee.Geometry.Polygon(
        [[[103.15527841413423, -2.8621137297832173],
          [103.15527841413423, -2.9077177848426454],
          [103.21124002302095, -2.9077177848426454],
          [103.21124002302095, -2.8621137297832173]]])

train5 = ee.Geometry.Polygon(
        [[[103.10892984235689, -2.8934881446424914],
          [103.10892984235689, -2.9373765773812446],
          [103.1755344566147, -2.9373765773812446],
          [103.1755344566147, -2.8934881446424914]]])

train6 =  ee.Geometry.Polygon(
        [[[103.17373201215669, -2.8252519817684183],
          [103.17373201215669, -2.9037746494754204],
          [103.26368257368013, -2.9037746494754204],
          [103.26368257368013, -2.8252519817684183]]])
train7 =  ee.Geometry.Polygon(
        [[[103.081034868602, -2.904460413137298],
          [103.081034868602, -2.9500627572273728],
          [103.13424989545747, -2.9500627572273728],
          [103.13424989545747, -2.904460413137298]]])
          
val1 = ee.Geometry.Polygon(
        [[[103.24273988569185, -2.8478835202194768],
          [103.24273988569185, -2.9483484170446874],
          [103.29046174848482, -2.9483484170446874],
          [103.29046174848482, -2.8478835202194768]]])
          
val2 = ee.Geometry.Polygon(
        [[[103.08240815961763, -2.8566274048086933],
          [103.08240815961763, -2.8988028504820584],
          [103.1301300224106, -2.8988028504820584],
          [103.1301300224106, -2.8566274048086933]]])

val3 =     ee.Geometry.Polygon(
        [[[103.15527841413423, -2.8609135984375555],
          [103.15527841413423, -2.8972598739369757],
          [103.21295663679048, -2.8972598739369757],
          [103.21295663679048, -2.8609135984375555]]])

train_poly = ee.FeatureCollection([ee.Feature(train1),
                                   ee.Feature(train2),
                                   ee.Feature(train3),
                                   ee.Feature(train4),
                                   ee.Feature(train5),
                                   ee.Feature(train6),
                                   ee.Feature(train7)])

val_poly =  ee.FeatureCollection([ee.Feature(val1),
                                   ee.Feature(val2),
                                  ee.Feature(val3)])

###########################################
# EXPORT
###########################################

#create a feature stack
if params['MODE'] == 'training':
  featureStack = ee.Image.cat([
                image.select(BANDS),
                label.select(RESPONSE_TR),
                mask.select(MASK)
                ])
else:
  featureStack = ee.Image.cat([
                image.select(BANDS),
                label.select(RESPONSE_TU)
                ])

list = ee.List.repeat(1, params['KERNEL_SIZE'])
lists = ee.List.repeat(list, params['KERNEL_SIZE'])
kernel = ee.Kernel.fixed(params['KERNEL_SIZE'], params['KERNEL_SIZE'], lists)

ARRAYS = featureStack.neighborhoodToArray(kernel)

#export dataset
helper.exportDataset(params, train_poly, val_poly, ARRAYS, FEATURES)
