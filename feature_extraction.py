import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureExtractionError(Exception):
    """Base class for feature extraction errors"""
    pass

class InvalidInputError(FeatureExtractionError):
    """Raised when input is invalid"""
    pass

class FeatureExtractionLayer(nn.Module):
    """
    Base class for feature extraction layers

    Attributes:
        input_shape (Tuple[int, int, int]): Input shape
        output_shape (Tuple[int, int, int]): Output shape
    """
    def __init__(self, input_shape: Tuple[int, int, int], output_shape: Tuple[int, int, int]):
        super(FeatureExtractionLayer, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class Conv2DLayer(FeatureExtractionLayer):
    """
    2D convolutional layer

    Attributes:
        input_shape (Tuple[int, int, int]): Input shape
        output_shape (Tuple[int, int, int]): Output shape
        num_filters (int): Number of filters
        kernel_size (int): Kernel size
    """
    def __init__(self, input_shape: Tuple[int, int, int], output_shape: Tuple[int, int, int], num_filters: int, kernel_size: int):
        super(Conv2DLayer, self).__init__(input_shape, output_shape)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(input_shape[0], num_filters, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.conv(x)
            return x
        except Exception as e:
            logging.error(f"Error in Conv2DLayer: {str(e)}")
            raise FeatureExtractionError(f"Error in Conv2DLayer: {str(e)}")

class MaxPool2DLayer(FeatureExtractionLayer):
    """
    2D max pooling layer

    Attributes:
        input_shape (Tuple[int, int, int]): Input shape
        output_shape (Tuple[int, int, int]): Output shape
        pool_size (int): Pool size
    """
    def __init__(self, input_shape: Tuple[int, int, int], output_shape: Tuple[int, int, int], pool_size: int):
        super(MaxPool2DLayer, self).__init__(input_shape, output_shape)
        self.pool_size = pool_size
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.max_pool(x)
            return x
        except Exception as e:
            logging.error(f"Error in MaxPool2DLayer: {str(e)}")
            raise FeatureExtractionError(f"Error in MaxPool2DLayer: {str(e)}")

class FlattenLayer(FeatureExtractionLayer):
    """
    Flatten layer

    Attributes:
        input_shape (Tuple[int, int, int]): Input shape
        output_shape (Tuple[int, int, int]): Output shape
    """
    def __init__(self, input_shape: Tuple[int, int, int], output_shape: Tuple[int, int, int]):
        super(FlattenLayer, self).__init__(input_shape, output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = x.view(-1, self.output_shape[0])
            return x
        except Exception as e:
            logging.error(f"Error in FlattenLayer: {str(e)}")
            raise FeatureExtractionError(f"Error in FlattenLayer: {str(e)}")

class DenseLayer(FeatureExtractionLayer):
    """
    Dense layer

    Attributes:
        input_shape (Tuple[int, int, int]): Input shape
        output_shape (Tuple[int, int, int]): Output shape
        num_units (int): Number of units
    """
    def __init__(self, input_shape: Tuple[int, int, int], output_shape: Tuple[int, int, int], num_units: int):
        super(DenseLayer, self).__init__(input_shape, output_shape)
        self.num_units = num_units
        self.dense = nn.Linear(input_shape[0], num_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.dense(x)
            return x
        except Exception as e:
            logging.error(f"Error in DenseLayer: {str(e)}")
            raise FeatureExtractionError(f"Error in DenseLayer: {str(e)}")

class FeatureExtractor(nn.Module):
    """
    Feature extractor

    Attributes:
        input_shape (Tuple[int, int, int]): Input shape
        output_shape (Tuple[int, int, int]): Output shape
        layers (List[FeatureExtractionLayer]): List of layers
    """
    def __init__(self, input_shape: Tuple[int, int, int], output_shape: Tuple[int, int, int]):
        super(FeatureExtractor, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = nn.ModuleList()

    def add_layer(self, layer: FeatureExtractionLayer):
        self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            for layer in self.layers:
                x = layer(x)
            return x
        except Exception as e:
            logging.error(f"Error in FeatureExtractor: {str(e)}")
            raise FeatureExtractionError(f"Error in FeatureExtractor: {str(e)}")

def create_feature_extractor(input_shape: Tuple[int, int, int], output_shape: Tuple[int, int, int]) -> FeatureExtractor:
    """
    Create a feature extractor

    Args:
        input_shape (Tuple[int, int, int]): Input shape
        output_shape (Tuple[int, int, int]): Output shape

    Returns:
        FeatureExtractor: Feature extractor
    """
    feature_extractor = FeatureExtractor(input_shape, output_shape)
    return feature_extractor

def add_conv2d_layer(feature_extractor: FeatureExtractor, num_filters: int, kernel_size: int) -> FeatureExtractor:
    """
    Add a 2D convolutional layer to the feature extractor

    Args:
        feature_extractor (FeatureExtractor): Feature extractor
        num_filters (int): Number of filters
        kernel_size (int): Kernel size

    Returns:
        FeatureExtractor: Feature extractor
    """
    input_shape = feature_extractor.input_shape
    output_shape = (num_filters, input_shape[1] - kernel_size + 1, input_shape[2] - kernel_size + 1)
    layer = Conv2DLayer(input_shape, output_shape, num_filters, kernel_size)
    feature_extractor.add_layer(layer)
    feature_extractor.input_shape = output_shape
    return feature_extractor

def add_maxpool2d_layer(feature_extractor: FeatureExtractor, pool_size: int) -> FeatureExtractor:
    """
    Add a 2D max pooling layer to the feature extractor

    Args:
        feature_extractor (FeatureExtractor): Feature extractor
        pool_size (int): Pool size

    Returns:
        FeatureExtractor: Feature extractor
    """
    input_shape = feature_extractor.input_shape
    output_shape = (input_shape[0], input_shape[1] // pool_size, input_shape[2] // pool_size)
    layer = MaxPool2DLayer(input_shape, output_shape, pool_size)
    feature_extractor.add_layer(layer)
    feature_extractor.input_shape = output_shape
    return feature_extractor

def add_flatten_layer(feature_extractor: FeatureExtractor) -> FeatureExtractor:
    """
    Add a flatten layer to the feature extractor

    Args:
        feature_extractor (FeatureExtractor): Feature extractor

    Returns:
        FeatureExtractor: Feature extractor
    """
    input_shape = feature_extractor.input_shape
    output_shape = (input_shape[0] * input_shape[1] * input_shape[2],)
    layer = FlattenLayer(input_shape, output_shape)
    feature_extractor.add_layer(layer)
    feature_extractor.input_shape = output_shape
    return feature_extractor

def add_dense_layer(feature_extractor: FeatureExtractor, num_units: int) -> FeatureExtractor:
    """
    Add a dense layer to the feature extractor

    Args:
        feature_extractor (FeatureExtractor): Feature extractor
        num_units (int): Number of units

    Returns:
        FeatureExtractor: Feature extractor
    """
    input_shape = feature_extractor.input_shape
    output_shape = (num_units,)
    layer = DenseLayer(input_shape, output_shape, num_units)
    feature_extractor.add_layer(layer)
    feature_extractor.input_shape = output_shape
    return feature_extractor

def main():
    # Create a feature extractor
    feature_extractor = create_feature_extractor((3, 224, 224), (1,))

    # Add layers to the feature extractor
    feature_extractor = add_conv2d_layer(feature_extractor, 64, 3)
    feature_extractor = add_maxpool2d_layer(feature_extractor, 2)
    feature_extractor = add_conv2d_layer(feature_extractor, 128, 3)
    feature_extractor = add_maxpool2d_layer(feature_extractor, 2)
    feature_extractor = add_flatten_layer(feature_extractor)
    feature_extractor = add_dense_layer(feature_extractor, 128)
    feature_extractor = add_dense_layer(feature_extractor, 1)

    # Print the feature extractor
    print(feature_extractor)

if __name__ == "__main__":
    main()