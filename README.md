# Image

A set of image classifiers on different datasets


# mnist-cnn

A convolutional neural net to train on the MNIST dataset.
the conv net is as follows: Conv->Relu->Pool->Conv->Relu->Pool->FC->FC

Max Pooling is used in the pooling layer
For the input layer, 28*28 monochromatic images are used from the mnist dataset
The classifier gives 86% accuracy on test data with 9000 training iterations 

# Face Detector

A small python program to detect faces in an image using Open CVs cascades.
To utilize, change the path to the folder or image you wish to detect faces in.
