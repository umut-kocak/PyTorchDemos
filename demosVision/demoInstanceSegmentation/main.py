"""
This script demonstrates using a pre-trained Mask R-CNN model for instance segmentation,
applying masks to images, and visualizing the results.
"""
from pathlib import Path
from typing import List

import numpy as np
import torch
from torchvision.io import read_image
from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_Weights,
                                          maskrcnn_resnet50_fpn)
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks, make_grid

from common.utils.arg_parser import get_args as get_common_args
from common.utils.visualise import display_grid

# Default values as constants
DEFAULT_IMAGE_FILE = "dog1.jpg"
DEFAULT_IMAGE_FILE2 = "dog2.jpg"

def get_args():
    """
    Parses command-line arguments and returns the configuration and arguments.

    Returns:
        args (Namespace): Parsed arguments, including configurations for the model URL, 
                          output model filename, and test image filenames.
    """
    parser, _ = get_common_args()
    parser.add_argument('--input-image-file-name', type=str, default=DEFAULT_IMAGE_FILE,
                        help='First image file to test the model with')
    parser.add_argument('--input-image-file-name2', type=str, default=DEFAULT_IMAGE_FILE2,
                        help='Second image file to test the model with')
    return parser.parse_args()

def load_and_transform_images(image_paths: List[str], transform, size : tuple=(400, 600)) -> List[torch.Tensor]:
    """
    Loads and transforms images from the specified paths.

    Args:
        image_paths (List[str]): List of image file paths.
        transform: The transformation function to apply to images.
        size tuple: The images are resized to the given size.

    Returns:
        List[torch.Tensor]: List of transformed images.
    """
    images = [read_image(path) for path in image_paths]
    transformed_images = [transform(img) for img in images]
    transformed_images = [v2.Resize(size=size)(orig_img) for orig_img in transformed_images]
    return transformed_images

def filter_and_draw_masks(images: List[torch.Tensor], model_outputs, 
                          categories: List[str], score_threshold: float = 0.75, 
                          proba_threshold: float = 0.5) -> List[torch.Tensor]:
    """
    Filters and draws masks on images based on score and probability thresholds.

    Args:
        images (List[torch.Tensor]): Original images.
        model_outputs: Model output containing masks, scores, and labels.
        categories (List[str]): List of category labels.
        score_threshold (float): Score threshold for mask filtering.
        proba_threshold (float): Probability threshold for converting masks to boolean.

    Returns:
        List[torch.Tensor]: Images with masks drawn.
    """
    bool_masks = [
        out['masks'][out['scores'] > score_threshold] > proba_threshold
        for out in model_outputs
    ]
    images_with_masks = [
        draw_segmentation_masks(img, mask.squeeze(1)) 
        for img, mask in zip(images, bool_masks)
    ]

    # Optional:
    for i, output in enumerate(model_outputs):
        print(f"Detected instances for the {i}th image:")
        print([categories[label] for label in output['labels']])
        print("Scores:", [score.item() for score in output['scores']])
    
    return images_with_masks

def main():
    """
    Main function that loads images, applies an instance segmentation model, 
    and displays the segmented images with masks.
    """
    args = get_args()
    torch.manual_seed(args.seed)
    
    # Load image paths
    image_paths = [
        str(Path(args.default_assets_path) / args.input_image_file_name),
        str(Path(args.default_assets_path) / args.input_image_file_name2)
    ]
    
    # Load model and transformations
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transform = weights.transforms()
    model = maskrcnn_resnet50_fpn(weights=weights, progress=False).eval()
    
    # Load and preprocess images
    images = load_and_transform_images(image_paths, transform)
    display_grid([make_grid(images)], title="Original Images")
    
    # Get model outputs
    with torch.no_grad():
        model_outputs = model(images)
    
    # Filter and draw masks based on thresholds
    categories = weights.meta["categories"]
    images_with_masks = filter_and_draw_masks(images, model_outputs, categories)
    
    # Plot images with drawn masks
    display_grid(images_with_masks, title="Images with Masks")

if __name__ == '__main__':
    main()
