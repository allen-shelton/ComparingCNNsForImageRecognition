# Comparing CNNs Architectures for Visual Impairment Assistive Technology Applications
Visual Impairment is a problem among people of all demographics, and much research has been done to offer people with visual impairment options for safely navigating their environment. This project builds upon work that I have already done in this area of research, where I developed a prototype for a deep learning-enabled guide in which a camera recognizes objects using a Convolutional Neural Network, and then a text-to-speech algorithm verbalizes what objects are identified. My project will focus on the CNN portion of that work, and trying to improve the prediction accuracy. The main things that were explored to improve the accuracy will be the collection of more data, the use of different CNN architectures for transfer learning, and utilizing data augmentation during training.

This repository contains all files used for this project, which include code to train 4 CNN architectures, namely AlexNet, VGG-16, GoogleNet, and ResNet50. Training was conducted both with and without data augmentation, and an additional test was conducted to test the effect of different data augmentation scenarios on validation accuracy with AlexNet. Additional helper functions for plotting graphical results is also included here. These files along with the images used to train the CNNs are all contained in the FinalProjectCode folder. The TrainingResults folder contains images of graphical results obtained from the best training run for all test cases. 

All code was written in MATLAB and requires the following toolboxes to run:
 - Deep Learning Toolbox
 - Deep Learning Toolbox Model for AlexNet Network
 - Deep Learning Toolbox Model for VGG-16 Network
 - Deep Learning Toolbox Model for GoogleNet Network
 - Deep Learning Toolbox Model for ResNet-50 Network
