# PyTorchDemos

# Repository Overview

This repository contains a collection of Python-based demos. The `demosBasic` folder includes demonstrations of core
functionalities, such as quantization, Automatic Mixed Precision (AMP), pruning, and exporting models in ONNX format.
The `demosVision` folder contains examples specific to computer vision, including object classification, detection,
instance segmentation, and semantic segmentation.

## Running the Demos

To run these demos successfully, follow these steps:

- **Create the Conda Environment**: Each demo folder contains a specific file for environment setup. Use this file to create a corresponding Conda environment.

  ```bash
  cd demosBasic
  conda create --name <env_name> --file conda_environment_specific.txt
  cd ..
  ```

Replace <env_name> with your preferred environment name.

- **Run the Demo in Module Mode**: After setting up the environment, run the demo from the root directory in module mode. For example:


  ```bash
  python -m demosBasic.demoDynamicQuantization.main
  ```
Ensure you are in the root directory before executing the Python command to run each demo.
