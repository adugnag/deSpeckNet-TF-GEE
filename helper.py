
import ee
import tensorflow as tf

###########################################
# 0. CONVERT to DB
###########################################

def lin_to_db(image):
    """
    Convert backscatter from linear to dB.
    Parameters
    ----------
    image : ee.Image
        Image to convert 
    Returns
    -------
    ee.Image
        output image
    """
    bandNames = image.bandNames().remove('angle')
    db = ee.Image.constant(10).multiply(image.select(bandNames).log10()).rename(bandNames)
    return image.addBands(db, None, True)

###########################################
# 1. PREPARE SENTINEL-1
###########################################

def s1_prep(params):
    """
    prepares S1 image collection based on a dictionary of parameters. 

    """
    
    POLARIZATION = params['POLARIZATION']
    FORMAT = params['FORMAT']
    START_DATE = params['START_DATE']
    STOP_DATE = params['STOP_DATE']
    ORBIT = params['ORBIT']
    RELATIVE_ORBIT_NUMBER = params['RELATIVE_ORBIT_NUMBER']
    ROI = params['ROI']
    CLIP_TO_ROI = params['CLIP_TO_ROI']
    
    if POLARIZATION is None: POLARIZATION = 'VVVH'
    if FORMAT is None: FORMAT = 'DB' 
    if ORBIT is None: ORBIT = 'DESCENDING' 
    
    
    pol_required = ['VV', 'VH', 'VVVH']
    if (POLARIZATION not in pol_required):
        raise ValueError("ERROR!!! Parameter POLARIZATION not correctly defined")

    
    orbit_required = ['ASCENDING', 'DESCENDING', 'BOTH']
    if (ORBIT not in orbit_required):
        raise ValueError("ERROR!!! Parameter ORBIT not correctly defined")


    format_required = ['LINEAR', 'DB']
    if (FORMAT not in format_required):
        raise ValueError("ERROR!!! FORMAT not correctly defined")
        

    
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                .filter(ee.Filter.eq('resolution_meters', 10)) \
                .filter(ee.Filter.eq('platform_number', 'A')) \
                .filterDate(START_DATE, STOP_DATE) \
                .filterBounds(ROI)


        # select orbit
    if (ORBIT != 'BOTH'):
      s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', ORBIT))

    if (RELATIVE_ORBIT_NUMBER != 'ANY'): 
      s1 =  s1.filter(ee.Filter.eq('relativeOrbitNumber_start', RELATIVE_ORBIT_NUMBER)) 
      
    
    if (POLARIZATION == 'VV'):
      s1 = s1.select(['VV','angle'])
    elif (POLARIZATION == 'VH'):
      s1 = s1.select(['VH','angle'])
    elif (POLARIZATION == 'VVVH'):
      s1 = s1.select(['VV','VH','angle'])  

    
    # clip image to roi
    if (CLIP_TO_ROI):
        s1 = s1.map(lambda image: image.clip(ROI))

    
    if (FORMAT == 'DB'):
        s1 = s1.map(lin_to_db)
        
        
    return s1

###########################################
# 2. EXPORT TRAIN AND VALIDATION DATA
###########################################

def exportDataset(params, train_poly, val_poly, arrays, FEATURES):
  trainingPolysList = train_poly.toList(train_poly.size())
  evalPolysList = val_poly.toList(val_poly.size())

  # These numbers determined experimentally.
  n = 250 # Number of shards in each polygon.
  N = 2500 # Total sample size in each polygon.

  # Export all the training data (in many pieces), with one task per geometry.
  for g in range(train_poly.size().getInfo()):
    geomSample = ee.FeatureCollection([])
    for i in range(n):
      sample = arrays.sample(
        region = ee.Feature(trainingPolysList.get(g)).geometry(), 
        scale = 10, 
        numPixels = N / n, # Size of the shard.
        seed = i,
        tileScale = 16
    )
    geomSample = geomSample.merge(sample)
  
    desc = params['TRAINING_BASE'] + '_g' + str(g)
    if params['EXPORT'] == 'GCS':
        task = ee.batch.Export.table.toCloudStorage(
            collection = geomSample,
            description = desc, 
            bucket = params['BUCKET'], 
            fileNamePrefix = params['FOLDER'] + '/' + desc,
            fileFormat = 'TFRecord',
            selectors = FEATURES
            )
    else:
        task = ee.batch.Export.table.toDrive(
            collection = geomSample,
            description = desc,  
            fileNamePrefix = params['FOLDER'] + '/' + desc,
            fileFormat = 'TFRecord',
            selectors = FEATURES
            )
    task.start()

  # Export all the evaluation data.
  for g in range(val_poly.size().getInfo()):
    geomSample = ee.FeatureCollection([])
    for i in range(n):
      sample = arrays.sample(
        region = ee.Feature(evalPolysList.get(g)).geometry(), 
        scale = 10, 
        numPixels = N / n,
        seed = i,
        tileScale = 16
      )
      geomSample = geomSample.merge(sample)
  
    desc = params['EVAL_BASE'] + '_g' + str(g)
    if params['EXPORT'] == 'GCS':
        task = ee.batch.Export.table.toCloudStorage(
            collection = geomSample,
            description = desc, 
            bucket = params['BUCKET'], 
            fileNamePrefix = params['FOLDER'] + '/' + desc,
            fileFormat = 'TFRecord',
            selectors = FEATURES
            )
    else:
        task = ee.batch.Export.table.toDrive(
            collection = geomSample,
            description = desc, 
            fileNamePrefix = params['FOLDER'] + '/' + desc,
            fileFormat = 'TFRecord',
            selectors = FEATURES
            )
    task.start()
    print('Exporting training and validation data')
    
###########################################
# 3. DATA PIPELINE
###########################################
    
#@title Helper functions
#simple data augmentation
class dataAugment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_masks = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels, masks):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    masks = self.augment_labels(masks)
    return inputs, (labels, inputs), masks


def parse_tfrecord(example_proto):
  return tf.io.parse_single_example(example_proto, FEATURES_DICT)


def to_tuple_train(inputs):
  inputsList = [inputs.get(key) for key in FEATURES]
  stacked = tf.stack(inputsList, axis=0)
  stacked = tf.transpose(stacked, [1, 2, 0])
  #select features
  data = stacked[:,:,:len(params['BANDS'])]
  #select labels
  if len(params['BANDS']) ==2:
      label = stacked[:,:,len(params['BANDS']):len(params['BANDS'])+2]
      masks = stacked[:,:,len(params['BANDS'])+2:len(params['BANDS'])+3]
  else:
      label = stacked[:,:,len(params['BANDS']):len(params['BANDS'])+1]
      masks = stacked[:,:,len(params['BANDS'])+1:len(params['BANDS'])+2]
  return data, label, masks

def to_tuple_tune(inputs):
  inputsList = [inputs.get(key) for key in FEATURES]
  stacked = tf.stack(inputsList, axis=0)
  stacked = tf.transpose(stacked, [1, 2, 0])
  data = stacked[:,:,:len(params['BANDS'])]
  #select features
  
  label = stacked[:,:,len(params['BANDS']):]
  return data, (label, data)

def get_dataset(pattern, params):
  glob = tf.io.gfile.glob(pattern)
  #glob =tf.compat.v1.gfile.Glob(pattern)
  dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
  dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
  if params['MODE'] == 'training':
      dataset = dataset.map(to_tuple_train, num_parallel_calls=5)
      dataset = dataset.map(dataAugment(), num_parallel_calls=5)
  else:
      dataset = dataset.map(to_tuple_tune, num_parallel_calls=5)
  return dataset


"""# Training data: use the tf.data api to build our data pipeline"""
def get_training_dataset(params, FEATURES, FEATURES_DICT):
    global params
    global FEATURES
    global FEATURES_DICT
    if params['EXPORT'] == 'GCS':
        glob = 'gs://' + params['BUCKET'] + '/' + params['FOLDER'] + '/' + params['TRAINING_BASE'] + '*'
    else:
        glob = params['DRIVE'] + '/' + params['FOLDER'] + '/' + params['TRAINING_BASE'] + '*'
    dataset = get_dataset(glob,params)
    dataset = dataset.shuffle(params['BUFFER_SIZE']).batch(params['BATCH_SIZE']).repeat()
    return dataset


def get_eval_dataset(params):
    if params['EXPORT'] == 'GCS':
        glob = 'gs://' + params['BUCKET'] + '/' + params['FOLDER'] + '/' + params['EVAL_BASE'] + '*'
    else:
        glob = params['DRIVE'] + '/' + params['FOLDER'] + '/' + params['EVAL_BASE'] + '*'
    dataset = get_dataset(glob,params)
    dataset = dataset.batch(1).repeat()
    return dataset


###########################################
# 4. MODEL
###########################################

def deSpeckNet(depth,filters,image_channels, use_bnorm=True):
    layer_count = 0
    inpt = tf.keras.layers.Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='glorot_normal', padding='same',use_bias = True,name = 'conv'+str(layer_count))(inpt)
    layer_count += 1
    x0 = tf.keras.layers.Activation('relu',name = 'relu'+str(layer_count))(x0)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth-2):
        layer_count += 1
        x0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='glorot_normal', padding='same',use_bias = True,name = 'conv'+str(layer_count))(x0)
        if use_bnorm:
            layer_count += 1
        x0 = tf.keras.layers.BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x0)
        layer_count += 1
        x0 = tf.keras.layers.Activation('relu',name = 'relu'+str(layer_count))(x0)  
    # last layer, Conv
    layer_count += 1
    x0 = tf.keras.layers.Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_normal',padding='same',use_bias = True,name = 'speckle'+str(1))(x0)
    layer_count += 1
    
    
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='glorot_normal', padding='same',use_bias = True,name = 'conv'+str(layer_count))(inpt)
    layer_count += 1
    x = tf.keras.layers.Activation('relu',name = 'relu'+str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth-2):
        layer_count += 1
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='glorot_normal', padding='same',use_bias = True,name = 'conv'+str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
        x = tf.keras.layers.BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        layer_count += 1
        x = tf.keras.layers.Activation('relu',name = 'relu'+str(layer_count))(x)  
    # last layer, Conv
    layer_count += 1
    x = tf.keras.layers.Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_normal',padding='same',use_bias = True,name = 'clean' + str(1))(x)
    layer_count += 1
    x_orig = tf.keras.layers.Add(name = 'noisy' +  str(1))([x0,x])
    
    model = tf.keras.Model(inputs=inpt, outputs=[x,x_orig])
    
    return model

#Learning rate scheduler
def lr_schedule(epoch):
    initial_lr = 1e-3
    if epoch<=30:
        lr = initial_lr
    elif epoch<=60:
        lr = initial_lr/10
    elif epoch<=80:
        lr = initial_lr/20 
    else:
        lr = initial_lr/20 
    tf.summary.scalar('learning rate', data=lr, step=epoch)
    return lr

#Total variation loss
def TVloss(y_true, y_pred):
  return tf.reduce_sum(tf.image.total_variation(y_pred))

###########################################
# 5. EXPORT AND PREDICT
###########################################

def export(image, params):
  """Run the image export task.  Block until complete.
  """
  task = ee.batch.Export.image.toCloudStorage(
    image = image.select(params['BANDS']),
    description = params['IMAGE_PREFIX'],
    bucket = params['BUCKET'],
    fileNamePrefix = params['FOLDER'] + '/' + params['IMAGE_PREFIX'],
    region = params['GEOMETRY'].getInfo()['coordinates'],
    scale = 10,
    fileFormat = 'TFRecord',
    maxPixels = 1e13,
    formatOptions = {
      'patchDimensions': params['KERNEL_SHAPE'],
      'kernelSize': params['KERNEL_BUFFER'],
      'compressed': True,
      'maxFileSize': 104857600
    }
  )
  task.start()

  # Block until the task completes.
  print('Running image export to Cloud Storage...')
  import time
  while task.active():
    time.sleep(30)

  # Error condition
  if task.status()['state'] != 'COMPLETED':
    print('Error with image export.')
  else:
    print('Image export completed.')


def prediction(params):
  """Perform inference on exported imagery, upload to Earth Engine.
  """

  print('Looking for TFRecord files...')

  # Get a list of all the files in the output bucket.
  filesList = !gsutil ls 'gs://'{params['BUCKET']}'/'{params['FOLDER']}

  # Get only the files generated by the image export.
  exportFilesList = [s for s in filesList if params['IMAGE_PREFIX'] in s]

  # Get the list of image files and the JSON mixer file.
  imageFilesList = []
  jsonFile = None
  for f in exportFilesList:
    if f.endswith('.tfrecord.gz'):
      imageFilesList.append(f)
    elif f.endswith('.json'):
      jsonFile = f

  # Make sure the files are in the right order.
  imageFilesList.sort()

  from pprint import pprint
  pprint(imageFilesList)
  print(jsonFile)

  import json
  # Load the contents of the mixer file to a JSON object.
  jsonText = !gsutil cat {jsonFile}
  # Get a single string w/ newlines from the IPython.utils.text.SList
  mixer = json.loads(jsonText.nlstr)
  pprint(mixer)
  patches = mixer['totalPatches']

  # Get set up for prediction.
  x_buffer = int(params['KERNEL_BUFFER'][0] / 2)
  y_buffer = int(params['KERNEL_BUFFER'][1] / 2)

  buffered_shape = [
      params['KERNEL_SHAPE'][0] + params['KERNEL_BUFFER'][0],
      params['KERNEL_SHAPE'][1] + params['KERNEL_BUFFER'][1]]

  imageColumns = [
    tf.io.FixedLenFeature(shape=buffered_shape, dtype=tf.float32) 
      for k in params['BANDS']
  ]

  imageFeaturesDict = dict(zip(params['BANDS'], imageColumns))

  def parse_image(example_proto):
    return tf.io.parse_single_example(example_proto, imageFeaturesDict)

  def toTupleImage(inputs):
    inputsList = [inputs.get(key) for key in params['BANDS']]
    stacked = tf.stack(inputsList, axis=0)
    stacked = tf.transpose(stacked, [1, 2, 0])
    #stacked = tf.reshape(tensor = stacked , shape = [NR_IMAGES, 32 , 32 ,len(BAND_MODE)])
    return stacked

   # Create a dataset from the TFRecord file(s) in Cloud Storage.
  imageDataset = tf.data.TFRecordDataset(imageFilesList, compression_type='GZIP')
  imageDataset = imageDataset.map(parse_image, num_parallel_calls=5)
  imageDataset = imageDataset.map(toTupleImage).batch(1)

  # Perform inference.
  print('Running predictions...')
  predictions = model.predict(imageDataset, steps=patches, verbose=1)
  predictions = predictions[0]
  #predictions = predictions.argmax(axis=3)
  print(len(predictions))
  print(predictions[0].shape)
 

  print('Writing predictions...')
  out_image_file = 'gs://' + params['BUCKET'] + '/' + params['FOLDER'] + '/' + params['MODEL_NAME'] + '.TFRecord'
  writer = tf.io.TFRecordWriter(out_image_file)
  patches = 0

  for predictionPatch in predictions:
    print('Writing patch ' + str(patches) + '...')
    predictionPatch = predictionPatch[
        x_buffer:x_buffer+params['KERNEL_SIZE'], y_buffer:y_buffer+params['KERNEL_SIZE'],:]

    if params['POLARIZATION'] == 'VVVH':
    # Create an example.
      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            'VV': tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=predictionPatch[:,:,0].flatten())),
            'VH': tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=predictionPatch[:,:,1].flatten()))
          }
        )
      )
    else:
      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            params['POLARIZATION']: tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=predictionPatch.flatten()))
          }
        )
      )
    # Write the example.
    writer.write(example.SerializeToString())
    patches += 1

  writer.close()

  # Start the upload.
  out_image_asset = USER_ID + '/' + params['MODEL_NAME']
  !earthengine upload image --asset_id={out_image_asset} {out_image_file} {jsonFile}
