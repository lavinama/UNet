# UNet
Segment salt deposits beneath the Earth's surface using seismic images.

This project was done for the [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)

### Table of Contents
1. [Background](#background)
2. [UNet Architecture](#unet-architecture)
3. [Training](#training)
4. [Dataset](#dataset)
5. [Results](#results)


### Background

In normal object detection networks we are interested *what* are the objects so we often use max pooling to reduce the size of the feature map. But with semantic segmentation we are also interested in ther *where* of the objects, so we use up sampling methods, such as transposed convolution.

What is transposed convolution?

It is exactly the opposite of normal convolution, the input is a low resolution image and the output is a high resolution image. To see how it works I invite you to check out this [post](https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0).

### UNet Architecture

The [UNet](https://arxiv.org/abs/1505.04597) architecture contains two paths:
1. The first path is the encoder, which its objective is to capture the context of the image. It is a traditional object detection network.
2. The second path is the decoder, which is used to precisely locate the objects.

We are getting $128 \times 128 \times 3$ images.

This is the architecture of the UNet:
<p align="center">
  <img src="https://github.com/lavinama/UNet/blob/main/media/unet_arch.jpeg", width=600 />
</p>

Advantages:
* The network is image size agnostic since it does not contain fully connected layers, this means that the model is of smaller weight size.
* Can be easily scaled ot multiple classes.
* Works well with small datasets, thanks to the robustness provided with data augmentation

Disadvantages:
* The size of the UNet should be similar to the size of the features (need context of the images).
* High number of layers means that it takes time to train.

There are other methods for semantic segmentation such as: FCN-VGG16, DeepLab, Deconvnet, U-Net, DialatedNet, GCN, PSPNet, FC-DenseNet103, EncNet, Gated-SCNN. For more information visit this [link](https://arxiv.org/pdf/2001.04074.pdf).

### Training

Optimisation techniques and Hyperparameters:
* Adam Optimiser
* Binary Cross Entropy Loss since we are dealing with only two classes, salt and no salt
* Learning rate decay if the validation loss does not improve for 5 continues epochs.
* Early stopping if the validation loss does not improve for 10 continues epochs.
* Batchsize is 32

### Dataset
You can download the dataset [here](https://www.kaggle.com/c/tgs-salt-identification-challenge/data)

The trainining directory `train.zip` contains 4000 seismic images together with 4000 grayscale images denoting where the salt deposits are. For example:

<p align="center">
  <img src="https://github.com/lavinama/UNet/blob/main/media/preview_data.png", width=400 />
</p>


### Results
Results on the validation dataset:
<p align="center">
  <img src="https://github.com/lavinama/UNet/blob/main/media/val1.png", width=400 /> </br>
  <img src="https://github.com/lavinama/UNet/blob/main/media/val2.png", width=400 /> </br>
  <img src="https://github.com/lavinama/UNet/blob/main/media/val3.png", width=400 /> </br>
  <img src="https://github.com/lavinama/UNet/blob/main/media/val4.png", width=400 /> </br>
  <img src="https://github.com/lavinama/UNet/blob/main/media/val5.png", width=400 /> </br>
</p>

The lowest validation loss that I got was 0.18702.
<p align="center">
  <img src="https://github.com/lavinama/UNet/blob/main/media/loss_curves.png", width=400 />
</p>

The results on the training dataset are better than those on the validation dataset -> model is overfitting.


This project was inspired by the work done by [Harshall Lamba](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47).
