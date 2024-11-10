from common.utils.arg_parser import get_args as get_common_args
from common.models import SuperResolutionNet

import onnx
import onnxruntime
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torchvision.transforms as transforms

import numpy as np
import time
from PIL import Image
import logging

# Default values as constants
DEFAULT_MODEL_URL = "https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth"
DEFAULT_IMAGE_FILE = "dog1.jpg"
DEFAULT_OUTPUT_MODEL_FILE = "super_resolution.onnx"

def get_args():
    """
    Parses command-line arguments and returns the configuration and arguments.

    Returns:
        args (Namespace): Parsed arguments, including configurations for the model URL, 
                          output model filename, and test image filename.
    """
    parser, config = get_common_args(True)

    parser.add_argument('--model-url', type=str, default=DEFAULT_MODEL_URL,
                        help='URL for the pretrained model weights')
    parser.add_argument('--output-model-file-name', type=str, default=DEFAULT_OUTPUT_MODEL_FILE,
                        help='Filename for the exported ONNX model')
    parser.add_argument('--test-image-file-name', type=str, default=DEFAULT_IMAGE_FILE,
                        help='Image file to test the model with')

    args = parser.parse_args()
    args.config = config
    return args

def load_model(model_url, upscale_factor=3):
    """
    Loads the SuperResolution model with pretrained weights.

    Args:
        model_url (str): URL to load model weights.
        upscale_factor (int): Factor by which to upscale the input image.

    Returns:
        torch.nn.Module: Super-resolution model with pretrained weights loaded.
    """
    model = SuperResolutionNet(upscale_factor=upscale_factor)
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))
    model.eval()
    return model

def export_to_onnx(model, output_file, input_shape, batch_size):
    """
    Exports the model to ONNX format.

    Args:
        model (torch.nn.Module): The super-resolution model.
        output_file (str): Output filename for the ONNX model.
        input_shape (tuple): Shape of the input tensor.
        batch_size (int): Batch size for the input tensor.
    """
    x = torch.randn(batch_size, *input_shape, requires_grad=True)
    torch.onnx.export(
        model, x, output_file, export_params=True, opset_version=10,
        do_constant_folding=True, input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    logging.info("Model exported successfully to ONNX format.")

def to_numpy(tensor):
    """
    Converts a tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): Tensor to convert.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def test_inference(model, ort_session, x):
    """
    Runs inference on both the PyTorch model and ONNX model, comparing performance.

    Args:
        model (torch.nn.Module): PyTorch model.
        ort_session (onnxruntime.InferenceSession): ONNX runtime session.
        x (torch.Tensor): Input tensor for inference.
    """
    # PyTorch inference timing
    start_time = time.time()
    torch_out = model(x)
    end_time = time.time()
    logging.info(f"PyTorch inference time: {end_time - start_time:.4f} seconds")

    # ONNX inference timing
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    start_time = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    end_time = time.time()
    logging.info(f"ONNX inference time: {end_time - start_time:.4f} seconds")

    # Result comparison
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    logging.info("ONNX and PyTorch results are consistent.")

def save_super_resolved_image(ort_session, img_y, cb, cr, output_filename):
    """
    Processes the ONNX output and saves the super-resolved image.

    Args:
        ort_session (onnxruntime.InferenceSession): ONNX session for inference.
        img_y (torch.Tensor): Luminance (Y) channel tensor of the input image.
        cb (Image): Original chroma (Cb) channel of the input image.
        cr (Image): Original chroma (Cr) channel of the input image.
        output_filename (str): Filename for saving the super-resolved image.

    Returns:
        np.ndarray: Size of the output image.
    """
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    img_out_y = ort_session.run(None, ort_inputs)[0]
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            cb.resize(img_out_y.size, Image.BICUBIC),
            cr.resize(img_out_y.size, Image.BICUBIC)
        ]).convert("RGB")
    
    final_img.save(output_filename)
    logging.info(f"Super-resolved image saved as {output_filename}.")
    return img_out_y.size

def main():
    """
    Main function that loads the model, exports it to ONNX, runs inference tests, 
    and saves a super-resolved image using the ONNX runtime.
    """
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    torch.manual_seed(args.seed)

    # Load and prepare model
    model = load_model(args.model_url)
    input_shape = (1, 224, 224)  # Single channel, 224x224 image
    export_to_onnx(model, args.output_model_file_name, input_shape, args.config.batch_size)

    # Create ONNX runtime session
    ort_session = onnxruntime.InferenceSession(args.output_model_file_name, providers=["CPUExecutionProvider"])
    x = torch.randn(args.config.batch_size, *input_shape, requires_grad=True)
    test_inference(model, ort_session, x)

    # Load and process test image
    img = Image.open(f"./assets/{args.test_image_file_name}")
    img = transforms.Resize([224, 224])(img)
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()
    img_y = transforms.ToTensor()(img_y).unsqueeze(0)

    # Save super-resolved image
    out_size = save_super_resolved_image(ort_session, img_y, img_cb, img_cr, f"./superres_{args.test_image_file_name}")
    
    # Save resized original image (without super-resolution)
    resized_img = transforms.Resize([out_size[0], out_size[1]])(img)
    output_filename = f"./resized_{args.test_image_file_name}" 
    resized_img.save(output_filename)
    logging.info(f"Resized image saved as {output_filename}.")
    
if __name__ == '__main__':
    main()
