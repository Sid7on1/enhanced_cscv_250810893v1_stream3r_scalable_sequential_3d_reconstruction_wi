import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Dict

# Define constants and configuration
DATA_ROOT = 'data'
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
NUM_WORKERS = 4

# Define exception classes
class DataLoaderError(Exception):
    """Base class for data loader exceptions."""
    pass

class InvalidDataError(DataLoaderError):
    """Raised when invalid data is encountered."""
    pass

# Define data structures/models
class ImageData:
    """Represents image data with its corresponding label."""
    def __init__(self, image: np.ndarray, label: int):
        self.image = image
        self.label = label

# Define validation functions
def validate_image_data(image: np.ndarray, label: int) -> bool:
    """Validates image data and its corresponding label."""
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return False
    if not isinstance(label, int):
        return False
    return True

# Define utility methods
def load_image(path: str) -> np.ndarray:
    """Loads an image from the given path."""
    try:
        image = np.load(path)
        return image
    except Exception as e:
        logging.error(f"Failed to load image: {e}")
        raise InvalidDataError(f"Failed to load image: {e}")

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocesses the image by normalizing its values."""
    image = image / 255.0
    return image

# Define the dataset class
class ImageDataset(Dataset):
    """Represents a dataset of images."""
    def __init__(self, data_root: str, transform: transforms.Compose):
        self.data_root = data_root
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self) -> None:
        """Loads the data from the data root directory."""
        for file in os.listdir(self.data_root):
            if file.endswith('.npy'):
                image_path = os.path.join(self.data_root, file)
                image = load_image(image_path)
                label = int(file.split('_')[0])
                if validate_image_data(image, label):
                    self.images.append(image)
                    self.labels.append(label)

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """Returns the image and its corresponding label at the given index."""
        image = self.images[index]
        label = self.labels[index]
        image = preprocess_image(image)
        image = self.transform(image)
        return image, label

# Define the data loader class
class ImageDataLoader:
    """Represents a data loader for images."""
    def __init__(self, data_root: str, batch_size: int, num_workers: int):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = ImageDataset(data_root, self.transform)
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)

    def get_data_loader(self) -> DataLoader:
        """Returns the data loader."""
        return self.data_loader

# Define the main class
class DataLoadingService:
    """Represents a service for loading image data."""
    def __init__(self, data_root: str, batch_size: int, num_workers: int):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_loader = ImageDataLoader(data_root, batch_size, num_workers)

    def load_data(self) -> DataLoader:
        """Loads the image data and returns the data loader."""
        return self.data_loader.get_data_loader()

# Define the main function
def main() -> None:
    """The main function."""
    data_root = DATA_ROOT
    batch_size = BATCH_SIZE
    num_workers = NUM_WORKERS
    data_loading_service = DataLoadingService(data_root, batch_size, num_workers)
    data_loader = data_loading_service.load_data()
    for batch in data_loader:
        images, labels = batch
        print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()