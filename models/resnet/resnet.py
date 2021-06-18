import torch
from torch import nn

from typing import Callable, List
from functools import partial

from .blocks import ConvNormAct, MaxBlurPool2d


def make_divisible(x, divisor=8):
    return int(max(1, x/divisor)*divisor)


class ResnetBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            bottleneck_factor: float = 0.5,
            activation: Callable[[], nn.Module] = nn.ReLU
    ):
        super().__init__()
        assert stride in {1, 2}

        hidden_channels = make_divisible(out_channels*bottleneck_factor)
        self.main_path = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation(),
            ConvNormAct(
                in_channels, hidden_channels,
                kernel_size=1, activation=activation
            ),
            ConvNormAct(
                hidden_channels, hidden_channels,
                kernel_size=3,
                stride=stride,
                activation=activation
            ),
            nn.Conv2d(
                hidden_channels, out_channels,
                kernel_size=1
            )
        )
        nn.init.constant_(self.main_path[-1].weight, 0)
        nn.init.constant_(self.main_path[-1].bias, 0)

        self.res_path = nn.Sequential(
            MaxBlurPool2d(
                kernel_size=3,
                blur_size=3,
                stride=2
            ) if stride > 1 else nn.Identity(),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1
            ) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.main_path(x) + self.res_path(x)


class ResnetStage(nn.Sequential):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int,
            stride: int = 1,
            bottleneck_factor: float = 0.5,
            activation: Callable[[], nn.Module] = nn.ReLU
    ):
        block = partial(
            ResnetBlock,
            bottleneck_factor=bottleneck_factor,
            activation=activation
        )

        super().__init__(
            block(in_channels, out_channels, stride=stride),
            *[
                block(out_channels, out_channels)
                for _ in range(n_blocks - 1)
            ],
        )


class Resnet(nn.Sequential):

    def __init__(
            self,
            n_blocks_list: List[int],
            input_channels: int = 3,
            base_channels: int = 64,
            bottleneck_factor: float = 0.5,
            activation: Callable[[], nn.Module] = nn.ReLU
    ):
        super().__init__()
        assert len(n_blocks_list) == 4
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.bottleneck_factor = bottleneck_factor
        stem_hidden_channels = make_divisible(base_channels/2)
        stem = nn.Sequential(
            ConvNormAct(
                input_channels, stem_hidden_channels,
                kernel_size=3,
                stride=2,
                activation=activation
            ),
            ConvNormAct(
                stem_hidden_channels, base_channels,
                kernel_size=3,
                activation=activation
            ),
            ConvNormAct(
                base_channels, base_channels,
                kernel_size=3,
                stride=2,
                normalization=None,
                activation=None,
            )
        )
        self.add_module('stem', stem)

        for idx, n_blocks in enumerate(n_blocks_list):
            self.add_module(
                f'layer_{idx+1}',
                ResnetStage(
                    make_divisible(base_channels*2**max(0, idx - 1)),
                    make_divisible(base_channels*2**max(0, idx)),
                    n_blocks=n_blocks,
                    stride=1 if idx == 0 else 2,
                    bottleneck_factor=bottleneck_factor,
                    activation=activation
                )
            )

        self.add_module(
            'final',
            ConvNormAct(
                self.out_channels, self.out_channels,
                kernel_size=3,
                activation=activation
            )
        )

    @property
    def out_channels(self):
        return self.base_channels*8
