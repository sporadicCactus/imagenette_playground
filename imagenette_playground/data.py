import torch

import pytorch_lightning as pl
import numpy as np
import cv2
import albumentations as A

from typing import Union, Callable
import os


class Datamodule(pl.LightningDataModule):

    def __init__(
            self,
            root: str,
            batch_size: int,
            num_workers: int,
            image_size: int,
            train_augmentations: Union[
                None,
                Callable[[np.ndarray], np.ndarray]
            ] = None,
        ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_augmentations = train_augmentations

        self.ds_train = FoldersAsClassesDataset(
            os.path.join(root, 'train'),
            transform=self.make_train_transform()
        )
        self.ds_val = FoldersAsClassesDataset(
            os.path.join(root, 'val'),
            transform=self.make_val_transform()
        )

        assert self.ds_train.n_classes == self.ds_val.n_classes

    @property
    def n_classes(self):
        return self.ds_train.n_classes

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_val,
            batch_size=3*self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def make_train_transform(self):
        alb_resize = A.RandomResizedCrop(
            self.image_size, self.image_size,
            scale=(0.7, 1.0),
            ratio=(9/11, 11/9),
        )
        def transform(image):
            image = alb_resize(image=image)['image']
            if self.train_augmentations:
                image = self.train_augmentations(image)
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float)
            return image
        return transform

    def make_val_transform(self):
        alb_resize = A.Sequential([
            A.SmallestMaxSize(self.image_size),
            A.CenterCrop(
                self.image_size, self.image_size
            )
        ])
        def transform(image):
            image = alb_resize(image=image)['image']
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float)
            return image
        return transform


class FoldersAsClassesDataset(torch.utils.data.Dataset):

    def __init__(
                self,
                root: str,
                transform: Union[
                    None,
                    Callable[[np.ndarray], np.ndarray]
                ] = None
        ):
        super().__init__()
        self.root = root
        self.transform = transform

        folders_with_images = {}

        for dirpath, _, filenames in os.walk(root):
            filenames = list(filter(
                lambda filename: filename.split('.')[-1].lower() in {
                    'jpg', 'jpeg', 'png'
                },
                filenames
            ))
            if filenames:
                folders_with_images[dirpath] = filenames

        self.dirpaths = sorted(folders_with_images.keys())

        samples = []
        for label, dirpath in enumerate(self.dirpaths):
            samples.extend([
                (
                    os.path.join(dirpath, filename),
                    label
                )
                for filename in folders_with_images[dirpath]
            ])
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = cv2.imread(path)

        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def n_classes(self):
        return len(self.dirpaths)
