import torch
from torch import nn

import pytorch_lightning as pl

import albumentations as A
import numpy as np

from functools import partial
from argparse import ArgumentParser

import optuna

from imagenette_playground.learner import Learner
from imagenette_playground.data import Datamodule
from models.resnet import Resnet


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--name', type=str, default='resnet_imagenette'
    )
    parser.add_argument(
        '--storage', type=str
    )
    parser.add_argument(
        '--n-epochs', type=int, default=5
    )
    parser.add_argument(
        '--image-size', type=int, default=129
    )
    parser.add_argument(
        '--dataset-path', type=str, default='imagenette'
    )
    parser.add_argument(
        '--n-trials', type=int, required=True
    )
    return parser.parse_args()


class DivergenceStoppingCallback(pl.callbacks.base.Callback):

    def __init__(self, monitor: str, consecutive_nans: int = 23):
        self.monitor = monitor
        self.consecutive_nans = consecutive_nans
        self._nan_count = 0

    def on_train_batch_end(self, trainer, *args, **kwargs):
        logs = trainer.callback_metrics
        monitor_val = logs.get(self.monitor)
        if isinstance(monitor_val, torch.Tensor):
            monitor_val = monitor_val.item()
        if monitor_val is None:
            return

        is_finite = np.isfinite(monitor_val)
        if not is_finite:
            self._nan_count += 1
        else:
            self._nan_count = 0

        if self._nan_count >= self.consecutive_nans:
            raise KeyboardInterrupt('Divergence condition met, interrupting.')


class Objective:

    def __init__(
            self,
            n_epochs: int,
            image_size: int,
            dataset_path: str,
        ):
        self.n_epochs = n_epochs
        self.dataset_path = dataset_path
        self.image_size = image_size

    def make_datamodule(self, trial):
        batch_size = trial.suggest_int(
            "batch_size",
            low=16, high=64, step=16
        )
        datamodule = Datamodule(
            self.dataset_path,
            batch_size=batch_size,
            num_workers=12,
            image_size=self.image_size,
            train_augmentations=lambda image: A.Sequential([
                A.HorizontalFlip(),
                A.CoarseDropout()
            ])(image=image)['image']
        )
        return datamodule

    def make_learner(self, trial, total_steps, n_classes):
        net = self.make_net(trial, n_classes)

        optimizer_type = trial.suggest_categorical(
            'optimizer',
            ['SGD', 'Adam']
        )
        if optimizer_type == 'SGD':
            optimizer_factory = partial(
                torch.optim.SGD,
                lr=0.0,
                weight_decay=1e-4
            )
            scheduler_factory = partial(
                torch.optim.lr_scheduler.OneCycleLR,
                max_lr=trial.suggest_loguniform(
                    "sgd_peak_lr",
                    low=1e-3, high=1e+0
                ),
                total_steps=total_steps
            )
        else:
            optimizer_factory = partial(
                torch.optim.Adam,
                lr=trial.suggest_loguniform(
                    "adam_lr",
                    low=1e-5, high=1e+0
                )
            )
            scheduler_factory = None

        learner = Learner(
            net,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory
        )
        return learner

    def make_net(self, trial, n_classes):
        net = Resnet(
            [
                trial.suggest_int(
                    f'layer_{idx+1}_blocks',
                    low=1, high=3
                )
                for idx in range(4)
            ],
            base_channels=trial.suggest_int(
                'base_channels',
                low=16, high=128, step=8
            )
        )
        net = nn.Sequential(
            net,
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(net.out_channels, n_classes)
        )
        return net

    def __call__(self, trial):
        datamodule = self.make_datamodule(trial)
        learner = self.make_learner(
            trial,
            len(datamodule.train_dataloader())*self.n_epochs + 1,
            datamodule.n_classes
        )
        trainer = pl.Trainer(
            logger=None,
            checkpoint_callback=None,
            gpus=1,
            precision=16,
            max_epochs=self.n_epochs,
            callbacks=[
                DivergenceStoppingCallback(monitor='train_loss')
            ]
        )
        trainer.fit(learner, datamodule=datamodule)
        metrics = trainer.test(learner, datamodule=datamodule)
        return metrics[0]['accuracy']


def main(args):
    study = optuna.create_study(
        study_name=args.name,
        storage=args.storage,
        load_if_exists=True,
        direction='maximize'
    )
    study.optimize(
        Objective(
            args.n_epochs,
            args.image_size,
            args.dataset_path,
        ),
        n_trials=args.n_trials
    )


if __name__ == '__main__':
    main(parse_args())
