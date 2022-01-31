# deSpeckNet-TF-GEE
 
This repository is the re-implementation of our paper [*deSpeckNet: Generalizing Deep Learning Based SAR Image Despeckling*](https://ieeexplore.ieee.org/document/9298453) published in IEEE Transactions on Geoscience and Remote Sensing. The original paper version of the code was implemented in Matlab but I think implementing the method in Tensorflow and Google Earth Engine (GEE) will improve its efficency and usabiltiy in the remote sensing community. The implementation uses python entirely and it seamlessly integrates image preparation in GEE with deep learning in Tensorflow.

**Note: I have made some modificatons from the original implementation, such as the data is processed in dB scale and patch density is different from the original Matlab implementation and the optimizer is Adam.**

## Architecture

It uses a simaese architecture to reconstruct the clean image and noise image simultaneously and reconstructing the original noisy image using two mean square error loss functions. To fine tune the model to new images with unknown speckle distribution, the model requires only the image to be despeckled.  
![drawing1](https://user-images.githubusercontent.com/48068921/102690422-96f76f00-4205-11eb-9ef0-5d98daecdee6.png)
![drawing_finetune](https://user-images.githubusercontent.com/48068921/102690424-99f25f80-4205-11eb-825b-dd9887935e67.png)

If interested, the pre-print version of the article is freely available [here](https://arxiv.org/pdf/2012.03066.pdf)

## Usage
 To train a model, the user needs to provide an area of interest in GEE geometry format and run the prepare_data.py first to prepare the training datasets. The user needs to select which mode to run the script on, either in training mode or tuning mode. The user needs to also specify their preference for storage of data as 'GCS' or 'Drive'. It is assumed the user have installed and configured Google cloud SDK on their local machine.
 
 To fine tune the model, the user needs to execute the prepare_data.py script one more time in tuning mode. Once a model is trained, the user can directly execute the test.py script to make inference on the selected data. By default, the despeckled image is uploaded to GEE. 
 
 A jupyter notebook version of the scripts is also included in the notebook folder, which should make it easier for users to run the code in Google colab without worrying about software dependencies. 

## Dependencies
To use the python scripts, we assume you have a gmail account and have already authenticated GEE and Cloud SDK on your local machine. The scripts are  written in Tensorflow 2.7 so there may be issues with earlier versions. To avoid these steps users could alternatively use the jupyter notebooks available in the notebooks folder to run the scripts in colab.

## Acknowledgment
Some IO functions were adopted from Google Earth Engine example workflow page. The page can be found [here](https://developers.google.com/earth-engine/guides/tf_examples).

## Reference

A. G. Mullissa, D. Marcos, D. Tuia, M. Herold and J. Reiche, "deSpeckNet: Generalizing Deep Learning-Based SAR Image Despeckling," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-15, 2022, Art no. 5200315, doi: 10.1109/TGRS.2020.3042694..
