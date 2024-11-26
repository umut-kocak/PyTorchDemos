"""
This script demonstrates keypoint detection on images using a pre-trained Keypoint R-CNN model,
visualizing detected keypoints with optional skeleton connectivity.
"""
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                          keypointrcnn_resnet50_fpn)
from torchvision.transforms import v2
from torchvision.utils import draw_keypoints

from common.utils.arg_parser import get_common_args
from common.utils.visualise import display_grid

# Default values as constants
DEFAULT_IMAGE_FILE = "skateboarder.jpg"
DEFAULT_DETECT_THRESHOLD = 0.75
DEFAULT_KEYPOINT_COLOR = "blue"
DEFAULT_KEYPOINT_RADIUS = 3
DEFAULT_CONNECT_RADIUS = 4
DEFAULT_CONNECT_WIDTH = 3

# Skeleton connections
DEFAULT_CONNECT_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
]


def get_args():
    """
    Parses command-line arguments and returns the configuration and arguments.
    Provides defaults for image file, detection threshold, keypoint color, radius, and connection parameters.

    Returns:
        argparse.Namespace: Parsed arguments with defaults or overridden values.
    """
    parser = get_common_args()

    # Image input
    parser.add_argument(
        '--input-image-file-name',
        type=str,
        default=DEFAULT_IMAGE_FILE,
        help='Path to the input image file to test the model with'
    )

    # Detection parameters
    parser.add_argument(
        '--detect-threshold',
        type=float,
        default=DEFAULT_DETECT_THRESHOLD,
        help='Detection threshold for filtering keypoints by score'
    )

    # Keypoint visualization parameters
    parser.add_argument(
        '--keypoint-color',
        type=str,
        default=DEFAULT_KEYPOINT_COLOR,
        help='Color for displaying keypoints'
    )
    parser.add_argument(
        '--keypoint-radius',
        type=int,
        default=DEFAULT_KEYPOINT_RADIUS,
        help='Radius for displaying individual keypoints'
    )

    # Connection visualization parameters
    parser.add_argument(
        '--connect-radius',
        type=int,
        default=DEFAULT_CONNECT_RADIUS,
        help='Radius for keypoints when connected as a skeleton'
    )
    parser.add_argument(
        '--connect-width',
        type=int,
        default=DEFAULT_CONNECT_WIDTH,
        help='Width of lines connecting keypoints in the skeleton visualization'
    )

    return parser.parse_args()


def load_and_transform_image(image_path, weights):
    """
    Loads an image and applies the model's required transformations.

    Args:
        image_path (str): Path to the image file.
        weights: Pretrained model weights containing the required transforms.

    Returns:
        torch.Tensor: Transformed image as a float tensor.
    """
    person_int = read_image(image_path)
    transforms = weights.transforms()
    return transforms(person_int), person_int


def detect_keypoints(model, image, threshold):
    """
    Detects keypoints in an image using the specified model and applies a score threshold.

    Args:
        model: The keypoint detection model.
        image (torch.Tensor): Transformed image tensor.
        threshold (float): Score threshold for filtering keypoints.

    Returns:
        torch.Tensor: Filtered keypoints based on the score threshold.
    """
    outputs = model([image])
    kpts = outputs[0]['keypoints']
    scores = outputs[0]['scores']
    idx = torch.where(scores > threshold)
    return kpts[idx]


def visualize_keypoints(image, keypoints, color, radius,
                        connect_skeleton=None, connect_width=None):
    """
    Visualizes keypoints on the input image with optional skeleton connectivity.

    Args:
        image (torch.Tensor): Original image tensor.
        keypoints (torch.Tensor): Filtered keypoints tensor.
        color (str): Color of keypoints.
        radius (int): Radius of keypoints.
        connect_skeleton (list of tuples, optional): List of keypoint pairs for connecting a skeleton.
        connect_width (int, optional): Width of lines connecting keypoints.
    """
    display_grid(
        [draw_keypoints(
            image, keypoints,
            connectivity=connect_skeleton,
            colors=color,
            radius=radius,
            width=connect_width if connect_skeleton else 1
        )], title="Keypoints"
    )


def main():
    """
    Main function to parse arguments, load the model and image, detect keypoints,
    and visualize the keypoints with optional skeleton connectivity.
    """
    args = get_args()
    torch.manual_seed(args.seed)

    # Load image and apply transformations
    image_path = str(
        Path(
            args.default_assets_path) /
        args.input_image_file_name)
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    person_float, person_int = load_and_transform_image(image_path, weights)

    # Initialize model and set to evaluation mode
    model = keypointrcnn_resnet50_fpn(weights=weights, progress=False)
    model.eval()

    # Detect keypoints
    keypoints = detect_keypoints(
        model, person_float, threshold=args.detect_threshold)

    # Visualize keypoints with skeleton connectivity
    visualize_keypoints(
        person_int, keypoints,
        color=args.keypoint_color,
        radius=args.connect_radius if args.connect_radius else args.keypoint_radius,
        connect_skeleton=DEFAULT_CONNECT_SKELETON,
        connect_width=args.connect_width
    )


if __name__ == '__main__':
    main()
