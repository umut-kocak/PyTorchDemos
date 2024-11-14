from common.utils.arg_parser import get_args as get_common_args

import torchvision
from torchvision import models, transforms
from captum.attr import Occlusion 
from captum.attr import visualization as viz

from PIL import Image
import requests
from io import BytesIO
import numpy as np
from pathlib import Path

# Default values as constants
DEFAULT_IMAGE_FILE = "cat_and_dog.jpg"
DEFAULT_DOG_TARGET = 208  # Labrador class in ImageNet
DEFAULT_CAT_TARGET = 283  # Persian cat class in ImageNet
DEFAULT_STRIDES = (3, 9, 9)
DEFAULT_SLIDING_WINDOW_SHAPES = (3, 45, 45)
DEFAULT_BASELINE = 0

def get_args():
    """
    Parses command-line arguments and returns the configuration and arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments with defaults or overridden values.
    """
    parser, _ = get_common_args()
    
    parser.add_argument(
        '--input-image-file-name', 
        type=str, 
        default=DEFAULT_IMAGE_FILE,
        help='Path to the input image file to test the model with'
    )
    return parser.parse_args()

def load_image(image_path):
    """
    Loads an image from the specified path.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        PIL.Image.Image: Loaded image.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        raise FileNotFoundError(f"Error loading image at {image_path}: {e}")

def preprocess_image(img):
    """
    Preprocesses the image for model input: resize, center crop, normalize.
    
    Args:
        img (PIL.Image.Image): Image to be preprocessed.
    
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(img).unsqueeze(0)

def compute_attribution(model, input_img, target, strides=DEFAULT_STRIDES, 
                        sliding_window_shapes=DEFAULT_SLIDING_WINDOW_SHAPES, baselines=DEFAULT_BASELINE):
    """
    Computes the occlusion-based attribution for a specified target.
    
    Args:
        model: Model to use for prediction and attribution.
        input_img (torch.Tensor): Preprocessed input image tensor.
        target (int): Target class index for which attribution is computed.
        strides (tuple): Stride dimensions for occlusion.
        sliding_window_shapes (tuple): Dimensions of the sliding window.
        baselines: Baseline values for occlusion.

    Returns:
        np.ndarray: Attribution tensor converted to a NumPy array.
    """
    occlusion = Occlusion(model)
    attribution = occlusion.attribute(
        input_img,
        strides=strides,
        target=target,
        sliding_window_shapes=sliding_window_shapes,
        baselines=baselines
    )
    return np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))

def visualize_attribution(attribution, original_img, title):
    """
    Visualizes the attribution results using Captum's visualization tools.
    
    Args:
        attribution (np.ndarray): Attribution data to visualize.
        original_img (PIL.Image.Image): Original image for overlaying attribution.
        title (str): Title for the visualization.
    """
    _ = viz.visualize_image_attr_multiple(
        attribution,
        np.array(original_img),
        ["heat_map", "original_image"],
        ["all", "all"],
        [title, "image"],
        show_colorbar=True
    )

def main():
    """
    Main function to parse arguments, load model and image, compute attributions for 
    different classes, and visualize the results.
    """
    args = get_args()
    image_path = str(Path(args.default_assets_path) / args.input_image_file_name)

    # Load and preprocess image
    img = load_image(image_path)
    input_img = preprocess_image(img)

    # Load model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()

    # Compute and visualize attribution for the dog class
    print("Calculating attribution for dog")
    attribution_dog = compute_attribution(model, input_img, target=DEFAULT_DOG_TARGET)
    visualize_attribution(attribution_dog, img, "Attribution for Dog")

    # Compute and visualize attribution for the cat class
    print("Calculating attribution for cat")
    attribution_cat = compute_attribution(model, input_img, target=DEFAULT_CAT_TARGET)
    visualize_attribution(attribution_cat, img, "Attribution for Cat")

if __name__ == '__main__':
    main()
