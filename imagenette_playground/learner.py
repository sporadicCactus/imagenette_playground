import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from tqdm import tqdm

from typing import Union, Callable, Iterable

from .losses import Loss, CrossEntropy
from .autoclip import AutoClip


class Learner(pl.LightningModule):

    def __init__(
            self,
            net: nn.Module,
            loss_fn: Loss = CrossEntropy(),
            optimizer_factory: Callable[
                [Iterable[nn.Parameter]],
                torch.optim.Optimizer
            ] = torch.optim.Adam,
            scheduler_factory: Union[
                None,
                Callable[[torch.optim.Optimizer], object]
            ] = None,
            grad_clip_percentile: int = 90
    ):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.autoclip = AutoClip(percentile=grad_clip_percentile)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        inputs, true_labels = batch
        pred_logits = self(inputs)
        loss = self.loss_fn(pred_logits, true_labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def _testval_step(self, batch):
        inputs, true_labels = batch
        pred_logits = self(inputs)
        losses = self.loss_fn(pred_logits, true_labels, reduction='none')
        pred_probs = F.softmax(pred_logits, dim=-1)
        return losses, pred_probs, true_labels

    def _testval_epoch_end(self, step_outputs):
        losses, pred_probs, true_labels = (torch.cat(x) for x in zip(*step_outputs))
        loss = losses.mean()
        accuracy = (pred_probs.max(dim=-1).indices == true_labels).float().mean()
        return {'loss': loss, 'accuracy': accuracy}

    def validation_step(self, batch, batch_idx):
        return self._testval_step(batch)

    def validation_epoch_end(self, step_outputs):
        metrics = self._testval_epoch_end(step_outputs)
        self.log_dict(
            {'val_'+name: value for name, value in metrics.items()},
            prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        return self._testval_step(batch)

    def test_epoch_end(self, step_outputs):
        metrics = self._testval_epoch_end(step_outputs)
        self.log_dict(metrics)

    def configure_optimizers(self):
        opt = self.optimizer_factory(self.parameters())
        if not self.scheduler_factory:
            return opt
        sch = self.scheduler_factory(opt)
        return [opt], [{
            'scheduler': sch,
            'interval': 'step',
            'frequency': 1
        }]

    def on_after_backward(self):
        grad_norm, clipped_grad_norm = self.autoclip(
            self.parameters()
        )

    def collect_batchnorm_statistics(self, batches):
        was_training = self.training
        _par = next(iter(self.parameters()))
        device, dtype = _par.device, _par.dtype
        self.train()
        batchnorm_momenta = {}
        for name, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm2d)):
                module.reset_running_stats()
                batchnorm_momenta[name] = module.momentum
                module.momentum = None

        for batch in tqdm(batches, desc='Collecting batchnorm statisctics'):
            inputs, _ = batch
            with torch.no_grad():
                self(inputs.to(device, dtype))

        for name, module in self.named_modules():
            if name in batchnorm_momenta:
                module.momentum = batchnorm_momenta[name]

        if not was_training:
            self.eval()
