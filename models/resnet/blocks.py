import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1,
            normalization=nn.BatchNorm2d,
            activation=nn.ReLU
        ):
        super().__init__()
        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups,
                padding=(kernel_size - 1)//2,
                bias=normalization is None,
            )
        )
        self.add_module(
            'norm',
            normalization(out_channels) if normalization is not None else nn.Identity()
        )
        self.add_module(
            'act',
            activation() if activation is not None else nn.Identity()
        )


class Blur2d(nn.Module):

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        base_kernel = torch.ones(2, 2)*0.5
        kernel = torch.ones(1, 1)
        for _ in range(kernel_size - 1):
            kernel = F.conv_transpose2d(
                kernel[None, None, ...],
                base_kernel[None, None, ...],
            ).squeeze()
        kernel = kernel/kernel.sum()
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        if self.kernel.shape == (1, 1) and self.stride == 1:
            return x
        x = F.conv2d(
            x,
            self.kernel[None, None, ...].expand(x.shape[1], 1, *self.kernel.shape),
            stride=self.stride,
            padding=(self.kernel_size - 1)//2,
            groups=x.shape[1]
        )
        return x


class MaxBlurPool2d(nn.Module):

    def __init__(self, kernel_size: int, blur_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.blur_size = blur_size
        self.stride = stride

        self.blur = Blur2d(blur_size, stride)

    def forward(self, x):
        if self.blur_size == 1:
            return F.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=(self.kernel_size - 1)//2
            )
        x = F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(self.kernel_size - 1)//2
        )
        x = self.blur(x)
        return x

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, blur_size={self.blur_size}, stride={self.stride}'
