# Semantic Segmentation with FCN-ResNet50

This script performs semantic segmentation on input images using a pre-trained Fully Convolutional Network (FCN) with a ResNet-50 backbone. The main features include:

1. **Image Preprocessing**: Input images are loaded, resized, and prepared for model processing.
2. **Semantic Segmentation**: The FCN-ResNet50 model predicts pixel-wise segmentation masks for different classes in the images.
3. **Class-Specific Mask Generation**: Masks are generated for user-specified target classes based on confidence thresholds.
4. **Visualization**: The segmentation masks are overlaid on the original images, allowing easy inspection of the model's predictions.

This demo highlights the application of deep learning in image segmentation, providing clear visualizations of class-specific and overall dominant masks.
