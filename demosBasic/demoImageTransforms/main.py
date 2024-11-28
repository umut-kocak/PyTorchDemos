"""
This script demonstrates applying various image transformations, including geometric,
photometric, and augmentation operations, using PyTorch and torchvision utilities.
"""
from pathlib import Path

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from common.utils.arg_parser import get_common_args
from common.utils.visualise import (display_grid_adapter,
                                    display_grid_with_annotations)

# Default values as constants
DEFAULT_IMAGE_FILE = "nature.jpg"


def get_args():
    """
    Parses command-line arguments for image transformation script.

    Returns:
        argparse.Namespace: Parsed arguments including the default image filename.
    """
    parser = get_common_args()
    parser.add_argument('--default-image-file-name', default=DEFAULT_IMAGE_FILE,
                        help='Default example image filename')
    return parser.parse_args()


def load_image(file_path):
    """
    Loads an image from the specified file path.

    Args:
        file_path (str): Path to the image file.

    Returns:
        Image.Image: Loaded PIL Image.
    """
    try:
        return Image.open(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None


def plot_transformations(orig_img, transformed_images, title=None, cmap=None):
    """
    Plots the original and transformed images in a grid format.

    Args:
        orig_img (Image.Image): The original image.
        transformed_images (list): List of transformed images to display.
        title (str, optional): Optional title for the plot.
        cmap (str, optional): Colormap to apply for grayscale images.
    """
    display_grid_adapter(
        [orig_img] +
        transformed_images,
        title=title,
        cmap=cmap)


def geometric_transforms(orig_img):
    """
    Applies a series of geometric transformations to the original image.

    Args:
        orig_img (Image.Image): The original image to transform.
    """
    print("Applying geometric transformations...")

    # Padding
    padded_imgs = [v2.Pad(padding)(orig_img) for padding in (3, 10, 30, 50)]
    plot_transformations(orig_img, padded_imgs, title="Padding")

    # Resize
    resized_imgs = [v2.Resize(size=size)(orig_img)
                    for size in (30, 50, 100, orig_img.size)]
    plot_transformations(orig_img, resized_imgs, title="Resize")

    # CenterCrop
    center_crops = [
        v2.CenterCrop(
            size=size)(orig_img) for size in (
            30,
            50,
            100,
            orig_img.size)]
    plot_transformations(orig_img, center_crops, title="CenterCrop")

    # FiveCrop
    top_left, top_right, bottom_left, bottom_right, center = v2.FiveCrop(
        size=(100, 100))(orig_img)
    plot_transformations(orig_img,
                         [top_left,
                          top_right,
                          bottom_left,
                          bottom_right,
                          center],
                         title="FiveCrop")

    # RandomPerspective
    perspective_transformer = v2.RandomPerspective(distortion_scale=0.6, p=1.0)
    perspective_imgs = [perspective_transformer(orig_img) for _ in range(4)]
    plot_transformations(orig_img, perspective_imgs, title="RandomPerspective")


def photometric_transforms(orig_img):
    """
    Applies a series of photometric transformations to the original image.

    Args:
        orig_img (Image.Image): The original image to transform.
    """
    print("Applying photometric transformations...")

    # Grayscale
    gray_img = v2.Grayscale()(orig_img)
    plot_transformations(orig_img, [gray_img], title="Grayscale", cmap='gray')

    # ColorJitter
    jitter = v2.ColorJitter(brightness=.5, hue=.3)
    jittered_imgs = [jitter(orig_img) for _ in range(4)]
    plot_transformations(orig_img, jittered_imgs, title="ColorJitter")

    # GaussianBlur
    blurrer = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
    blurred_imgs = [blurrer(orig_img) for _ in range(4)]
    plot_transformations(orig_img, blurred_imgs, title="GaussianBlur")


def augmentation_transforms(orig_img):
    """
    Applies a series of augmentation transforms to the original image.

    Args:
        orig_img (Image.Image): The original image to transform.
    """
    print("Applying augmentation transformations...")

    # AutoAugment
    policies = [
        v2.AutoAugmentPolicy.CIFAR10,
        v2.AutoAugmentPolicy.IMAGENET,
        v2.AutoAugmentPolicy.SVHN]
    augmenters = [v2.AutoAugment(policy) for policy in policies]
    imgs = [
        [augmenter(orig_img) for _ in range(4)]
        for augmenter in augmenters
    ]
    row_title = [str(policy).rsplit('.', maxsplit=1)[-1] for policy in policies]
    display_grid_adapter([[orig_img] + row for row in imgs],
                         title="AutoAugment Policies", row_title=row_title)

    # RandAugment
    augmenter = v2.RandAugment()
    imgs = [augmenter(orig_img) for _ in range(4)]
    plot_transformations(orig_img, imgs, title="RandAugment")


def example_preprocess(orig_img):
    """
    Demonstrates a sample preprocessing pipeline for an image.

    Args:
        orig_img (Image.Image): Original image to transform with bounding boxes.
    """
    print("Applying example preprocessing...")

    # Define bounding boxes
    boxes = tv_tensors.BoundingBoxes(
        [[15, 10, 370, 510], [275, 340, 510, 510], [130, 345, 210, 425]],
        format="XYXY", canvas_size=orig_img.size
    )

    # Define transformation pipeline
    transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
    ])

    # Apply transform
    out_img, out_boxes = transform(orig_img, boxes)
    display_grid_with_annotations(
        [(orig_img, boxes), (out_img, out_boxes)], title="Example Preprocessing with Bounding Boxes")


def main():
    """
    Main function to load an image and apply various transformations.

    The function demonstrates geometric, photometric, and augmentation transformations
    on the specified image and visualizes the results.
    """
    args = get_args()
    torch.manual_seed(args.seed)

    # Load image
    file_path = Path(args.default_assets_path) / args.default_image_file_name
    orig_img = load_image(file_path)
    if orig_img is None:
        return

    # Apply transformations
    geometric_transforms(orig_img)
    photometric_transforms(orig_img)
    augmentation_transforms(orig_img)
    example_preprocess(orig_img)


if __name__ == '__main__':
    main()
