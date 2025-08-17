import logging
import os
from typing import Dict, Any

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigException(Exception):
    """Base exception class for configuration-related errors."""
    pass

class InvalidConfigError(ConfigException):
    """Raised when the configuration is invalid."""
    pass

class Config:
    """
    Model configuration class.

    This class provides a centralized way to manage model configuration settings.
    It includes methods for loading, saving, and validating configuration data.
    """

    def __init__(self, config_file: str = 'config.json'):
        """
        Initializes the Config class.

        Args:
        - config_file (str): The path to the configuration file. Defaults to 'config.json'.
        """
        self.config_file = config_file
        self.config_data: Dict[str, Any] = {}

    def load_config(self) -> None:
        """
        Loads the configuration data from the specified file.

        Raises:
        - InvalidConfigError: If the configuration file is invalid or cannot be loaded.
        """
        try:
            import json
            with open(self.config_file, 'r') as file:
                self.config_data = json.load(file)
        except FileNotFoundError:
            logger.warning(f"Configuration file '{self.config_file}' not found. Using default settings.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid configuration file: {e}")
            raise InvalidConfigError("Invalid configuration file")

    def save_config(self) -> None:
        """
        Saves the configuration data to the specified file.

        Raises:
        - InvalidConfigError: If the configuration file cannot be saved.
        """
        try:
            import json
            with open(self.config_file, 'w') as file:
                json.dump(self.config_data, file, indent=4)
        except Exception as e:
            logger.error(f"Failed to save configuration file: {e}")
            raise InvalidConfigError("Failed to save configuration file")

    def get_config(self, key: str) -> Any:
        """
        Retrieves a configuration value by key.

        Args:
        - key (str): The configuration key.

        Returns:
        - The configuration value associated with the key.

        Raises:
        - KeyError: If the key is not found in the configuration data.
        """
        if key not in self.config_data:
            raise KeyError(f"Configuration key '{key}' not found")
        return self.config_data[key]

    def set_config(self, key: str, value: Any) -> None:
        """
        Sets a configuration value by key.

        Args:
        - key (str): The configuration key.
        - value (Any): The configuration value.

        Raises:
        - InvalidConfigError: If the configuration key or value is invalid.
        """
        if not isinstance(key, str):
            raise InvalidConfigError("Invalid configuration key")
        self.config_data[key] = value

    def validate_config(self) -> None:
        """
        Validates the configuration data.

        Raises:
        - InvalidConfigError: If the configuration data is invalid.
        """
        # Implement configuration validation logic here
        pass

class ModelConfig(Config):
    """
    Model configuration class.

    This class extends the Config class and provides model-specific configuration settings.
    """

    def __init__(self, config_file: str = 'model_config.json'):
        super().__init__(config_file)
        self.model_settings: Dict[str, Any] = {}

    def load_model_config(self) -> None:
        """
        Loads the model configuration data from the specified file.

        Raises:
        - InvalidConfigError: If the model configuration file is invalid or cannot be loaded.
        """
        try:
            import json
            with open(self.config_file, 'r') as file:
                self.model_settings = json.load(file)
        except FileNotFoundError:
            logger.warning(f"Model configuration file '{self.config_file}' not found. Using default settings.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid model configuration file: {e}")
            raise InvalidConfigError("Invalid model configuration file")

    def save_model_config(self) -> None:
        """
        Saves the model configuration data to the specified file.

        Raises:
        - InvalidConfigError: If the model configuration file cannot be saved.
        """
        try:
            import json
            with open(self.config_file, 'w') as file:
                json.dump(self.model_settings, file, indent=4)
        except Exception as e:
            logger.error(f"Failed to save model configuration file: {e}")
            raise InvalidConfigError("Failed to save model configuration file")

    def get_model_config(self, key: str) -> Any:
        """
        Retrieves a model configuration value by key.

        Args:
        - key (str): The model configuration key.

        Returns:
        - The model configuration value associated with the key.

        Raises:
        - KeyError: If the key is not found in the model configuration data.
        """
        if key not in self.model_settings:
            raise KeyError(f"Model configuration key '{key}' not found")
        return self.model_settings[key]

    def set_model_config(self, key: str, value: Any) -> None:
        """
        Sets a model configuration value by key.

        Args:
        - key (str): The model configuration key.
        - value (Any): The model configuration value.

        Raises:
        - InvalidConfigError: If the model configuration key or value is invalid.
        """
        if not isinstance(key, str):
            raise InvalidConfigError("Invalid model configuration key")
        self.model_settings[key] = value

class TrainingConfig(ModelConfig):
    """
    Training configuration class.

    This class extends the ModelConfig class and provides training-specific configuration settings.
    """

    def __init__(self, config_file: str = 'training_config.json'):
        super().__init__(config_file)
        self.training_settings: Dict[str, Any] = {}

    def load_training_config(self) -> None:
        """
        Loads the training configuration data from the specified file.

        Raises:
        - InvalidConfigError: If the training configuration file is invalid or cannot be loaded.
        """
        try:
            import json
            with open(self.config_file, 'r') as file:
                self.training_settings = json.load(file)
        except FileNotFoundError:
            logger.warning(f"Training configuration file '{self.config_file}' not found. Using default settings.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid training configuration file: {e}")
            raise InvalidConfigError("Invalid training configuration file")

    def save_training_config(self) -> None:
        """
        Saves the training configuration data to the specified file.

        Raises:
        - InvalidConfigError: If the training configuration file cannot be saved.
        """
        try:
            import json
            with open(self.config_file, 'w') as file:
                json.dump(self.training_settings, file, indent=4)
        except Exception as e:
            logger.error(f"Failed to save training configuration file: {e}")
            raise InvalidConfigError("Failed to save training configuration file")

    def get_training_config(self, key: str) -> Any:
        """
        Retrieves a training configuration value by key.

        Args:
        - key (str): The training configuration key.

        Returns:
        - The training configuration value associated with the key.

        Raises:
        - KeyError: If the key is not found in the training configuration data.
        """
        if key not in self.training_settings:
            raise KeyError(f"Training configuration key '{key}' not found")
        return self.training_settings[key]

    def set_training_config(self, key: str, value: Any) -> None:
        """
        Sets a training configuration value by key.

        Args:
        - key (str): The training configuration key.
        - value (Any): The training configuration value.

        Raises:
        - InvalidConfigError: If the training configuration key or value is invalid.
        """
        if not isinstance(key, str):
            raise InvalidConfigError("Invalid training configuration key")
        self.training_settings[key] = value

def main():
    # Create a Config instance
    config = Config('config.json')
    config.load_config()

    # Create a ModelConfig instance
    model_config = ModelConfig('model_config.json')
    model_config.load_model_config()

    # Create a TrainingConfig instance
    training_config = TrainingConfig('training_config.json')
    training_config.load_training_config()

    # Get configuration values
    print(config.get_config('key'))
    print(model_config.get_model_config('key'))
    print(training_config.get_training_config('key'))

    # Set configuration values
    config.set_config('key', 'value')
    model_config.set_model_config('key', 'value')
    training_config.set_training_config('key', 'value')

    # Save configuration files
    config.save_config()
    model_config.save_model_config()
    training_config.save_training_config()

if __name__ == '__main__':
    main()