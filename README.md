# Torch-7 on multiple GPUs over CNN


This is an example of how to use Torch-7 in a cluster with multiple NVIDIA GPUs to create a Convolutional neural netework.

I used fbcunn from Facebook AI Research,for installation detail you can refer at this link:

https://github.com/facebook/fbcunn/blob/master/INSTALL.md

Before using this code should be changed [config.lua] to set the paths 

The code is tested with CIFAR-10 dataset, it Consists of 60000 32x32 color images in 10 classes, with 6000 images per class.

