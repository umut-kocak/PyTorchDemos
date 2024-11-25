"""
This script demonstrates object detection using Faster R-CNN on two input images,
visualizing the detected bounding boxes.
"""
from pathlib import Path
from typing import List

import numpy as np
import torch
from torchvision.io import read_image
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                          fasterrcnn_resnet50_fpn)
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes, make_grid

from common.utils.arg_parser import get_args as get_common_args
from common.utils.visualise import display_grid

# Default values as constants
DEFAULT_IMAGE_FILE = "dog1.jpg"
DEFAULT_IMAGE_FILE2 = "dog2.jpg"


def get_args():
    """
    Parses command-line arguments and returns the configuration and arguments.

    Returns:
        Namespace: Parsed arguments, including configurations for the model URL,
                   output model filename, and test image filenames.
    """
    parser, _ = get_common_args()
    parser.add_argument('--input-image-file-name', type=str, default=DEFAULT_IMAGE_FILE,
                        help='First image file to test the model with')
    parser.add_argument('--input-image-file-name2', type=str, default=DEFAULT_IMAGE_FILE2,
                        help='Second image file to test the model with')
    return parser.parse_args()


def load_images(image_paths: List[str], size: tuple = (
        400, 600)) -> List[torch.Tensor]:
    """
    Loads and preprocesses images from specified paths.

    Args:
        image_paths (List[str]): List of image file paths.
        size tuple: The images are resized to the given size.

    Returns:
        List[torch.Tensor]: List of transformed images.
    """
    images = [read_image(path) for path in image_paths]
    images = [v2.Resize(size=size)(orig_img) for orig_img in images]
    return images


def main():

    args = get_args()
    torch.manual_seed(args.seed)

    # Load images
    image_paths = [
        str(Path(args.default_assets_path) / args.input_image_file_name),
        str(Path(args.default_assets_path) / args.input_image_file_name2)
    ]
    images = load_images(image_paths)
    display_grid([make_grid(images)], title="Original Images")

    # Load model and transformations
    # Use FasterRCNN to detect the bounding boxes
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    transform = weights.transforms()
    model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
    model = model.eval()

    # Run model
    batch = torch.stack([transform(img) for img in images])
    with torch.no_grad():
        outputs = model(batch)
    # print(outputs)

    # Plot the bounding boxes detedted by FasterRCNN with ascore greater than
    # a given threshold.
    score_threshold = .8
    dogs_with_boxes = [
        draw_bounding_boxes(
            dog_int, boxes=output['boxes'][output['scores'] > score_threshold], width=4)
        for dog_int, output in zip(images, outputs)
    ]
    display_grid(dogs_with_boxes, title="Detections")


if __name__ == '__main__':
    main()
