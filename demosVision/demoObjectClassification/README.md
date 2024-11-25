# CIFAR-10 Image Classification Demo

This script showcases the implementation of a custom image classification network trained on the CIFAR-10 dataset. Key features of the script include:

1. **Data Preparation**: CIFAR-10 training and testing datasets are downloaded, preprocessed, and loaded into PyTorch data loaders with configurable options for batch size and device compatibility.
2. **Model Initialization**: A custom classification network is defined, including convolutional and fully connected layers optimized for CIFAR-10's 10-class image dataset.
3. **Training and Evaluation**: The model is trained using stochastic gradient descent (SGD) and evaluated on the test dataset.
4. **Sample Prediction**: After training, the script performs a sample prediction to demonstrate the model's ability to classify CIFAR-10 images.

The demo provides a full workflow from dataset preparation to model training and inference, offering a simple and customizable introduction to image classification in PyTorch.
