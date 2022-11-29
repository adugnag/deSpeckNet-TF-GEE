# deSpeckNet-TF-GEE
 
This repository contains the re-implementation of our paper [*deSpeckNet: Generalizing Deep Learning Based SAR Image Despeckling*](https://ieeexplore.ieee.org/document/9298453) published in *IEEE Transactions on Geoscience and Remote Sensing*. The original paper version of the code was implemented in Matlab but I think implementing the method in Tensorflow and Google Earth Engine (GEE) will improve its usabiltiy in the remote sensing community. The implementation uses python and seamlessly integrates Sentinel-1 SAR image preparation in GEE with deep learning in Tensorflow.

**Note: I have made some modificatons from the original implementation, such as the data is processed in dB scale, patch density is different from the original Matlab implementation and the optimizer is Adam. I have also implemented multi-channel despeckling and data augmentation, which were not available in the original version.**

## Architecture

deSpeckNet uses a siamese architecture to reconstruct the clean image and the original noisy image using two mean square error loss functions. To fine tune the model to new images with unknown speckle distribution, the model *does not* require any clean reference image.  
![drawing1](https://user-images.githubusercontent.com/48068921/102690422-96f76f00-4205-11eb-9ef0-5d98daecdee6.png)
![drawing_finetune](https://user-images.githubusercontent.com/48068921/102690424-99f25f80-4205-11eb-825b-dd9887935e67.png)

If interested, the pre-print version of the article is freely available [here](https://arxiv.org/pdf/2012.03066.pdf)

## Usage
 To train a model, the user needs to provide an area of interest in GEE geometry format and run the prepare_data.py first to prepare the training datasets. The user needs to select training mode to run the script. The user needs to also specify their preference for storage of data as 'GCS' or 'Drive'. It is assumed the user have installed and configured Google cloud SDK on their local machine. For users that prefer to use google drive, the drive should be mounted at /content/drive for the scripts to run. 
 
 A jupyter notebook version of the scripts is also included in the notebook folder, which should make it easier for users to run the code in Google colab without worrying about software dependencies. 

## Dependencies
The scripts are  written in Tensorflow 2.7 so there may be issues with earlier versions of Tensorflow. 

## Acknowledgment
Some functions were adopted from Google Earth Engine example workflow [page](https://developers.google.com/earth-engine/guides/tf_examples).

## Reference

A. G. Mullissa, D. Marcos, D. Tuia, M. Herold and J. Reiche, "deSpeckNet: Generalizing Deep Learning-Based SAR Image Despeckling," in *IEEE Transactions on Geoscience and Remote Sensing*, vol. 60, pp. 1-15, 2022, Art no. 5200315, doi: 10.1109/TGRS.2020.3042694.
