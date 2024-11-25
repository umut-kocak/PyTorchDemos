"""
This script demonstrates semantic segmentation using a pre-trained FCN-ResNet50 model,
visualizing class-specific masks and overlaying them on images.
"""
from pathlib import Path
from typing import List

import numpy as np
import torch
from torchvision.io import read_image
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
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
        Namespace: Parsed arguments, including configurations for the model URL,
                   output model filename, and test image filenames.
    """
    parser, _ = get_common_args()
    parser.add_argument('--input-image-file-name', type=str, default=DEFAULT_IMAGE_FILE,
                        help='First image file to test the model with')
    parser.add_argument('--input-image-file-name2', type=str, default=DEFAULT_IMAGE_FILE2,
                        help='Second image file to test the model with')
    return parser.parse_args()


def load_images(image_paths: List[str], size : tuple=(400, 600)) -> List[torch.Tensor]:
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

def generate_boolean_masks(output, categories: List[str], target_classes: List[str], proba_threshold: float = 0.5):
    """
    Generates boolean masks for specified classes from model output.

    Args:
        output (torch.Tensor): Model output tensor of shape (batch_size, num_classes, H, W).
        categories (List[str]): List of category labels.
        target_classes (List[str]): Classes for which masks should be generated.
        proba_threshold (float): Probability threshold to convert masks to boolean.

    Returns:
        List[torch.Tensor]: List of boolean masks for each target class.
    """
    class_indices = {cls: idx for idx, cls in enumerate(categories)}
    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    # Generate and return boolean masks for each target class
    return [
        normalized_masks[:, class_indices[cls], :, :] > proba_threshold
        for cls in target_classes
    ]


def draw_class_masks_on_images(images: List[torch.Tensor], boolean_masks: List[torch.Tensor], alpha: float = 0.6) -> List[torch.Tensor]:
    """
    Draws segmentation masks on images.

    Args:
        images (List[torch.Tensor]): Original images.
        boolean_masks (List[torch.Tensor]): Boolean masks to overlay on images.
        alpha (float): Transparency level for masks.

    Returns:
        List[torch.Tensor]: Images with overlaid masks.
    """
    return [
        draw_segmentation_masks(img, mask, alpha=alpha)
        for img, mask in zip(images, boolean_masks)
    ]


def main():
    """
    Main function to load images, apply segmentation model, and visualize images with segmentation masks.
    """
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
    weights = FCN_ResNet50_Weights.DEFAULT
    transform = weights.transforms(resize_size=None)
    model = fcn_resnet50(weights=weights, progress=False).eval()

    # Run model and get segmentation output
    batch = torch.stack([transform(img) for img in images])
    with torch.no_grad():
        output = model(batch)['out']

    # Generate and visualize masks for specified classes
    target_classes = ['dog', 'boat']
    categories = weights.meta["categories"]
    class_masks = generate_boolean_masks(output, categories, target_classes)

    # Draw and plot masks on images for each specified class
    for idx, cls in enumerate(target_classes):
        print(f"Displaying masks for '{cls}' class:")
        masks_on_images = draw_class_masks_on_images(images, class_masks[idx])
        display_grid([masks_on_images], title=f"Masks for {cls}")

    # Generate and display masks for all classes
    print("Displaying masks for all classes:")
    num_classes = output.shape[1]
    all_class_masks = (output.argmax(dim=1) == torch.arange(num_classes)[:, None, None, None]).swapaxes(0, 1)
    masks_for_all_classes = draw_class_masks_on_images(images, all_class_masks)
    display_grid(masks_for_all_classes, title="Dominating mask")


if __name__ == '__main__':
    main()
