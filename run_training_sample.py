import torch
from torch import nn

import pytorch_lightning as pl
import albumentations as A

from functools import partial

from torchvision.models import resnet18

from imagenette_playground.learner import Learner
from imagenette_playground.data import Datamodule
from models.resnet_fpn import ResnetBackbone, ConvNormAct
from models.resnet import Resnet


class KeepLast(nn.Module):
    
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return x[-1]
        return x


def main():
    n_epochs = 5

    net = Resnet([2, 2, 2, 2], base_channels=96, activation=nn.Hardswish)
    net = nn.Sequential(
        net,
        nn.AdaptiveMaxPool2d(1),
        nn.Flatten(),
        nn.Linear(net.out_channels, 10)
    )
    nn.init.constant_(net[-1].weight, 0)
    nn.init.constant_(net[-1].bias, 0)

    datamodule = Datamodule(
        'imagenette', batch_size=32, num_workers=12, image_size=129,
        train_augmentations=lambda image: A.Sequential([
            A.HorizontalFlip(),
            A.CoarseDropout()
        ])(image=image)['image']
    )

    learner = Learner(
        net,
        optimizer_factory=partial(torch.optim.SGD, lr=0.0, weight_decay=1e-4),
        scheduler_factory=partial(
            torch.optim.lr_scheduler.OneCycleLR,
            max_lr=1e-2,
            total_steps=len(datamodule.train_dataloader())*n_epochs + 1,
        )
    )

    trainer = pl.Trainer(
        logger=None,
        checkpoint_callback=None,
        max_epochs=n_epochs,
        gpus=1,
        precision=16
    )

    trainer.fit(learner, datamodule=datamodule)

    metrics = trainer.test(learner, datamodule=datamodule)


if __name__ == '__main__':
    main()
