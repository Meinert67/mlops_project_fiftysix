from pathlib import Path
import typer
from torch.utils.data import Dataset
import os
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torch
from loguru import logger
import warnings

# Suppress warnings (torch load disable Weights_only=True warning)
warnings.filterwarnings("ignore")

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.load_raw_data(self.data_path)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.train_set)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.train_set[index]

    def load_raw_data(self, raw_data_path):
        # Downloads cifar-10 raw data if it doesn't already exist
        self.raw_train_set = datasets.CIFAR10(root=raw_data_path, train=True, download=True, transform=ToTensor())
        self.raw_test_set = datasets.CIFAR10(root=raw_data_path, train=False, download=True, transform=ToTensor())

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

        # Load train_set if it exists, else process and save it
        train_set_path = os.path.join(output_folder, 'train_set.pt')
        if os.path.exists(train_set_path):
            logger.info("train_set.pt exists. Loading data...")
            self.train_set = torch.load(train_set_path)
        else:
            logger.info("Processing raw train data and saving...")
            self.train_set = self.raw_train_set
            torch.save(self.raw_train_set, train_set_path)

        # Load test_set if it exists, else process and save it
        test_set_path = os.path.join(output_folder, 'test_set.pt')
        if os.path.exists(test_set_path):
            logger.info("test_set.pt exists. Loading data...")
            self.test_set = torch.load(test_set_path)
        else:
            logger.info("Processing raw test data and saving...")
            self.test_set = self.raw_test_set
            torch.save(self.raw_test_set, test_set_path)

default_raw_path = Path(__file__).parent.parent.parent / 'data/raw'
default_pro_path = Path(__file__).parent.parent.parent / 'data/processed'

def preprocess(raw_data_path: Path = default_raw_path, output_folder: Path = default_pro_path) -> None:
    """
    Starts the processes of the dataset
    Args:
    raw_data_path (Path): Where the raw data should be saved
    output_folder (Path): Where the processed data should be saved
    Returns:
        Class: MyDataset  
    """
    logger.info(f"Preparing data processing \n from: {raw_data_path} \n to: {output_folder}")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)
    
    return dataset


if __name__ == "__main__":
    #typer.run(preprocess)
    preprocess()
