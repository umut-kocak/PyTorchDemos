# demosVision

This repository contains a collection of PyTorch demos illustrating various deep learning techniques and tasks. Each demo is contained within its own subfolder and showcases specific functionalities, ranging from basic image classification to advanced neural network concepts. The purpose of these demos is to serve as learning resources and practical examples for working with PyTorch in diverse machine learning applications.

## Demos Overview

### 1. demoCaptum
Demonstrates how to use a pre-trained ResNet model with Captum to compute and visualize occlusion-based attributions for image classification, highlighting model interpretability techniques.

### 2. demoInstanceSegmentation
Showcases a pre-trained Mask R-CNN model to perform instance segmentation on images, overlaying detected object masks and visualizing results.

### 3. demoKeypoint
Uses a pre-trained Keypoint R-CNN model to detect and visualize keypoints in images, more specifically pose estimation where specific points on an object (such as joints on a human body) are identified in an image. 

### 4. demoObjectClassification
Illustrates training and evaluation of a custom classification network on the CIFAR-10 dataset, with a sample prediction included.

### 5. demoObjectClassificationFashion
A variant of object classification training and evaluating a custom classification network on the FashionMNIST dataset, including a sample prediction.

### 6. demoObjectDetection
Demonstrates object detection using Faster R-CNN, including visualization of bounding boxes around objects on two sample images.

### 7. demoSemanticSegmentation
Covers semantic segmentation using a pre-trained FCN-ResNet50 model with visualization of class-specific and dominant segmentation masks.

### 8. demoSpatialTransformer
Showcases a Spatial Transformer Network (STN), a differentiable module that enables the network to focus on relevant regions of an image, trained on the MNIST dataset, featuring dynamic spatial transformations and visualization.

### 9. demoTransferLearning
Highlights transfer learning, where a pre-trained ResNet18 model is fine-tuned for classification on the Hymenoptera dataset.

