# FashionMNIST Image Classification Demo

This script showcases the implementation of a custom classification network trained on the FashionMNIST dataset. The script provides a complete workflow for image classification, including:

1. **Data Preparation**: The FashionMNIST dataset is downloaded and preprocessed using PyTorch's `DataLoader`. Both training and testing data loaders are configured for efficient batch processing on the selected device (CPU/GPU).
2. **Model Initialization**: A custom classification network is created, tailored to handle grayscale images (1 channel) of size 28x28 and classify them into one of 10 fashion categories.
3. **Training and Evaluation**: The model is trained using stochastic gradient descent (SGD) and cross-entropy loss, then evaluated on the test dataset to measure its accuracy.
4. **Sample Prediction**: After training, the script performs a sample prediction on an image from the test dataset, printing the predicted and actual class labels.

This demo provides a simple and adaptable example of deep learning for image classification using PyTorch.
