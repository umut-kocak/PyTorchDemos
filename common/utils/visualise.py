import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image

plt.rcParams["savefig.bbox"] = 'tight'

def convert_image_np(tensor, mean=None, std=None):
    """
    Convert a PyTorch Tensor to a numpy image for visualization.

    Args:
        tensor (torch.Tensor): Image tensor in CHW format.
        mean (list, optional): Mean values for un-normalizing the tensor.
        std (list, optional): Standard deviation values for un-normalizing the tensor.

    Returns:
        numpy.ndarray: Converted numpy image in HWC format.
    """
    img = tensor.cpu().numpy().transpose((1, 2, 0))  # Convert to HWC
    if mean is not None and std is not None:
        mean = np.array(mean)
        std = np.array(std)
        img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

def display_image(image, title=None, mean=None, std=None, skip_show=False, **imshow_kwargs):
    """
    Display a single image using matplotlib.

    Args:
        image (torch.Tensor or numpy.ndarray): Image in CHW (Tensor) or HWC (numpy) format.
        title (str, optional): Title of the image.
        mean (list, optional): Mean values for un-normalizing the tensor.
        std (list, optional): Standard deviation values for un-normalizing the tensor.
        skip_show (bool, optional): If True, the image will not be shown (useful for testing).
        **imshow_kwargs: Additional keyword arguments to pass to plt.imshow.
    """
    if isinstance(image, torch.Tensor):
        image = convert_image_np(image, mean, std)

    plt.imshow(image, **imshow_kwargs)
    if title:
        plt.title(title)
    plt.axis('off')
    if not skip_show:
        plt.show()

def display_grid(images, title=None, row_title=None, num_cols=None, **imshow_kwargs):
    """
    Display a grid of images.

    Args:
        images (list of torch.Tensor or list of list of torch.Tensor): 1D or 2D list of images in CHW format.
        num_cols (int, optional): Number of columns to use for a 1D list of images.
        row_title (list of str, optional): Titles for each row.
        **imshow_kwargs: Additional keyword arguments to pass to plt.imshow.
    """
    # Convert 1D list to a 2D grid if necessary
    if not isinstance(images[0], list):
        # If images is a 1D list, create a 2D list with specified number of columns
        if num_cols is None:
            num_cols = len(images)  # Default to one row if no num_cols is given
        images = [images[i:i + num_cols] for i in range(0, len(images), num_cols)]
    
    # Determine grid size
    num_rows = len(images)
    num_cols = max(len(row) for row in images)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)

    # Plot each image
    for row_idx, row in enumerate(images):
        for col_idx, img in enumerate(row):
            if img is not None:
                img = img.detach()  # Detach the image from computation graph if needed
                img = F.to_pil_image(img)  # Convert tensor to PIL image for display
                axs[row_idx, col_idx].imshow(img, **imshow_kwargs)
            axs[row_idx, col_idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        # Add row titles if provided
        if row_title is not None and row_idx < len(row_title):
            axs[row_idx, 0].set_ylabel(row_title[row_idx])

    # Hide any empty subplots (in case of non-square grids)
    for ax in axs.flat:
        if not ax.has_data():
            ax.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def convert_pil_to_tensor(image):
    """
    Convert a PIL Image to a PyTorch tensor in CHW format.
    
    Args:
        image (PIL.Image.Image): The PIL image to convert.
    
    Returns:
        torch.Tensor: Image tensor in CHW format.
    """
    return TF.to_tensor(image)

def display_image_adapter(image, *args, **kwargs):
    """
    Adapter for display_image to handle PIL images.
    
    Args:
        image (torch.Tensor or PIL.Image.Image): Image to display.
        *args, **kwargs: Additional arguments to pass to display_image.
    """
    if isinstance(image, Image.Image):
        image = convert_pil_to_tensor(image)
    display_image(image, *args, **kwargs)

def display_grid_adapter(images, *args, **kwargs):
    """
    Adapter for display_grid to handle lists of PIL images.
    
    Args:
        images (list of torch.Tensor or PIL.Image.Image): List of images to display.
        *args, **kwargs: Additional arguments to pass to display_grid.
    """
    images = [convert_pil_to_tensor(img) if isinstance(img, Image.Image) else img for img in images]
    display_grid(images, *args, **kwargs)


def display_grid_with_annotations(images_with_boxes, title=None):
    """
    Display a grid of images with optional bounding boxes.

    Args:
        images_with_boxes (list of tuples): List of tuples where each tuple contains:
                                            - image (PIL.Image or torch.Tensor)
                                            - boxes (BoundingBoxes or torch.Tensor)
        title (str, optional): Title of the plot.
    """
    # Prepare lists for images and boxes
    images = []
    boxes = []
    
    for img, box in images_with_boxes:
        images.append(img)
        boxes.append(box)

    # Convert PIL images to tensors if necessary
    images = [convert_pil_to_tensor(img) if isinstance(img, Image.Image) else img for img in images]

    # Create the plot
    fig, axs = plt.subplots(1, len(images), squeeze=False)

    for i, (img, box) in enumerate(zip(images, boxes)):
        img = img.detach()  # Detach from computation graph if it's a tensor
        
        # Draw bounding boxes if provided
        if box is not None:
            img = draw_bounding_boxes(img, box, colors="yellow", width=3)

        img = F.to_pil_image(img)  # Convert tensor back to PIL for displaying
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    if title:
        plt.suptitle(title)
    plt.show()
