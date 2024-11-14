# demosVision

This repository contains a collection of PyTorch demos illustrating various deep learning techniques and tasks. Each demo is contained within its own subfolder and showcases specific functionalities, ranging from basic image classification to advanced neural network concepts. The purpose of these demos is to serve as learning resources and practical examples for working with PyTorch in diverse machine learning applications.

## Demos Overview

### 1. demoCaptum
Demonstrates the use of **Captum**, a library for model interpretability in PyTorch. This demo likely includes examples of visualizing model predictions, understanding feature importance, and using attribution techniques such as Integrated Gradients, Saliency, and Layer Conductance to interpret model behavior.

### 2. demoInstanceSegmentation
Showcases an **instance segmentation** task, where each individual object in an image is segmented separately. This demo may use models such as Mask R-CNN, commonly employed for detecting and delineating multiple objects within images with pixel-level precision for each instance.

### 3. demoKeypoint
Focuses on **keypoint detection**, a task often applied in pose estimation where specific points on an object (such as joints on a human body) are identified in an image. This demo could involve training or using models that detect these key points for applications in human pose estimation or object part localization.

### 4. demoObjectClassification
Illustrates **object classification** using deep neural networks. This task involves classifying entire images into predefined categories (e.g., identifying objects like cats, dogs, or vehicles in images). This demo is likely based on popular architectures such as ResNet, VGG, or MobileNet.

### 5. demoObjectClassificationFashion
A variant of object classification focusing specifically on **fashion items**. This demo may use datasets such as Fashion-MNIST or custom datasets of clothing categories to classify items like shirts, dresses, shoes, etc., which could be useful in retail or e-commerce applications.

### 6. demoObjectDetection
Demonstrates **object detection**, a task that combines localization and classification by identifying and drawing bounding boxes around objects in an image. This demo might utilize models like Faster R-CNN, YOLO, or SSD, popular choices for real-time object detection tasks.

### 7. demoSemanticSegmentation
Covers **semantic segmentation**, a pixel-level classification task where each pixel in an image is labeled as belonging to a particular class (e.g., road, sky, person). Unlike instance segmentation, semantic segmentation does not differentiate between different instances of the same class. Models used here could include Fully Convolutional Networks (FCNs) or U-Net.

### 8. demoSpatialTransformer
This demo likely showcases the **Spatial Transformer Network (STN)**, a differentiable module that enables the network to focus on relevant regions of an image, learning spatial transformations like rotation, scaling, and cropping. STNs can improve model accuracy in tasks requiring alignment or attention to specific parts of an image.

### 9. demoTransferLearning
Highlights **transfer learning**, where a pre-trained model is adapted to a new task with minimal training. This demo could involve fine-tuning a model like ResNet or VGG on a custom dataset, demonstrating the efficiency of transfer learning for tasks with limited data.
