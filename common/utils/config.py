"""
Configuration Management Module.

This module provides a `Config` class for managing configuration parameters used in machine
learning or deep learning applications. It supports default configuration values and the
ability to load and save configurations using JSON files. The class also includes methods
for validation and customization of configuration parameters.

Key Features:
- Default configurations with standard parameters such as batch size, learning rate, and epochs.
- Load configurations from a JSON file to override defaults.
- Save the current configuration to a JSON file for reproducibility.
- Automatic validation of configuration parameters to ensure correctness.

Classes:
    - Config: A class to manage and validate application configurations.

Functions:
    - load_from_json: Load configuration parameters from a JSON file.
    - save_to_json: Save the current configuration to a JSON file.
    - _validate_config: Validate configuration values to meet basic constraints.

Usage:
    - Initialize `Config` with or without a JSON file to use default or customized parameters.
    - Save the current configuration to a file for later use.
    - Override specific parameters dynamically after loading from a file or using defaults.

Example:
    config = Config("DefaultConfig.json")
    print(config)
    config.save_to_json("SavedConfig.json")
"""
import json


class Config:
    """
    Initializes the Config class with default values. Optionally loads configurations
    from a JSON file if provided.

    Args:
        config_file (str, optional): Path to a JSON file containing configuration
                                     parameters. If not provided, default values are used.
    """

    def __init__(self, config_file: str = None):
        # Default configuration values
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 1
        self.learning_rate = 0.001
        self.learning_rate_gamma = 0.7
        self.dry_run = False
        self.log_to_tensorboard = False
        self.log_interval = 10
        self.momentum = 0.0

        # Load configuration from JSON file if provided
        if config_file:
            self.load_from_json(config_file)

        # Validate loaded or default configuration values
        self._validate_config()

    def load_from_json(self, config_file: str):
        """
        Loads configuration parameters from a JSON file, overwriting default values.

        Args:
            config_file (str): Path to the JSON file containing configuration parameters.

        Raises:
            FileNotFoundError: If the specified JSON file is not found.
            json.JSONDecodeError: If the JSON file is not valid.
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Set attributes dynamically based on the JSON keys
            for key, value in config_data.items():
                setattr(self, key, value)

            # Re-validate after loading JSON data
            self._validate_config()

        except FileNotFoundError:
            print(
                f"Warning: Config file '{config_file}' not found. Using default values.")
        except json.JSONDecodeError as e:
            print(
                f"Error: Config file '{config_file}' is not a valid JSON. Using default values.")
            print(f"{e.msg} at line {e.lineno}, column {e.colno}")

    def save_to_json(self, config_file: str):
        """
        Saves the current configuration to a JSON file.

        Args:
            config_file (str): Path to the JSON file where the configuration should be saved.
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(self.__dict__, f, indent=4)
            print(f"Configuration saved to '{config_file}' successfully.")
        except IOError as e:
            print(f"Error: Unable to save config to '{config_file}'. {e}")

    def _validate_config(self):
        """
        Validates configuration values to ensure they meet basic constraints, such as
        positive values where appropriate.

        Raises:
            ValueError: If any configuration parameter fails validation.
        """
        if not (isinstance(self.batch_size, int) and self.batch_size > 0):
            raise ValueError("Batch size must be a positive integer.")
        if not (isinstance(self.test_batch_size, int)
                and self.test_batch_size > 0):
            raise ValueError("Test batch size must be a positive integer.")
        if not (isinstance(self.epochs, int) and self.epochs > 0):
            raise ValueError("Epochs must be a positive integer.")
        if not (isinstance(self.learning_rate, float)
                and self.learning_rate > 0):
            raise ValueError("Learning rate must be a positive float.")
        if not (isinstance(self.learning_rate_gamma, float)
                and 0 < self.learning_rate_gamma <= 1):
            raise ValueError(
                "Learning rate gamma must be a float between 0 and 1.")
        if not (isinstance(self.log_interval, int) and self.log_interval > 0):
            raise ValueError("Log interval must be a positive integer.")

    def __repr__(self):
        """
        Returns a string representation of the Config object, displaying the current
        configuration parameters and their values.

        Returns:
            str: String representation of the Config object.
        """
        return (f"Config(batch_size={self.batch_size}, test_batch_size={self.test_batch_size}, "
                f"epochs={self.epochs}, learning_rate={self.learning_rate}, "
                f"learning_rate_gamma={
            self.learning_rate_gamma}, dry_run={
            self.dry_run}, "
            f"log_to_tensorboard={self.log_to_tensorboard}, log_interval={self.log_interval})")


# Usage example:
# config = Config("DefaultConfig.json")
# print(config)
# config.save_to_json("SavedConfig.json")
