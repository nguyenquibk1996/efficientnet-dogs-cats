import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DogsAndCatsDataset(Dataset):
    def __init__(
        self, 
        img_dir: str,
        csv_file: str,
    ):
        '''
        Args:
            root_dir: Directory with all images.
            csv_dir: Path to the csv file with annotations.
        '''
        self.img_dir = img_dir
        self.annotations_csv = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        '''
        Args:
            idx: item's index
        Output:
            (image, label) 
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_path = os.path.join(self.img_dir, self.annotations_csv.iloc[idx, 1])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print('Failed to read image')
            return image_path
        
        label = self.annotations_csv.iloc[idx, 2]
        results = {'image': torch.tensor(image), 'label': torch.tensor(label)}
        
        return results


def get_dataloaders(
    train_csv_path: str,
    val_csv_path: str,
    img_dir: str,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    num_workers: int = 4,
):
    '''
    Args
        train_csv_path: path to the csv file of the training dataset.
        val_csv_path: path to the csv file of the validation dataset.
        train_batch_size: the batch size of the training dataloader.
        val_batch_size: the batch size of the validation dataloader.
        num_workers: the number of workers is used to load data.
    Output
        the dataloaders for the training and validation sets. 
    '''
    train_dataset = DogsAndCatsDataset(
        img_dir=img_dir,
        csv_file=train_csv_path,
    )

    val_dataset = DogsAndCatsDataset(
        img_dir=img_dir,
        csv_file=val_csv_path,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader
