# Occlusion-Based Attribution Visualization

This script demonstrates how to use a pre-trained ResNet model and the Captum library to compute and visualize occlusion-based attributions for image classification. The script performs the following tasks:

1. Loads an input image and preprocesses it for a ResNet model.
2. Uses Captum's `Occlusion` method to compute attribution maps for specified target classes (e.g., dog and cat classes in ImageNet).
3. Visualizes the computed attributions as heatmaps overlaid on the original image using Captum's visualization tools.

The purpose is to illustrate how model predictions are influenced by different parts of the input image, providing insights into the model's decision-making process.
