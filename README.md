# PyTorchDemos

# Repository Overview

This repository contains a collection of Python-based demos. The `demosBasic` folder includes demonstrations of core
functionalities, such as quantization, Automatic Mixed Precision (AMP), pruning, and exporting models in ONNX format.
The `demosVision` folder contains examples specific to computer vision, including object classification, detection,
instance segmentation, and semantic segmentation. Some common functionality is also gathered under the `common` folder.

## Running the Demos

To run these demos successfully, follow these steps:

- **Create the Conda Environment**: Use the `requirements_conda` file to create a corresponding Conda environment.

  ```bash
  conda env create -f ./requirements_conda.yaml --prefix ./env
  conda activate ./env
  ```

- **Run the Demo in Module Mode**: After setting up the environment, run the demo from the root directory in module mode. For example:


  ```bash
  python -m demosBasic.demoDynamicQuantization.main
  ```
Ensure you are in the root directory before executing the Python command to run each demo.

- **View Help Messages**: To display the help message with input parameters for a specific demo, use the `-h` option as shown below:


  ```bash
  python -m demosBasic.demoDynamicQuantization.main -h
  ```
