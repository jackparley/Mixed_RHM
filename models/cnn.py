import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1d_mixed_start_2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include a bias term
        """
        super().__init__()
        a=0.1
        self.filter_size_2 = 2
        self.filter_size_3 = 3
        self.stride = 5  # Stride should be the sum of both patch sizes (2+3)

        # Two separate filters
        self.filter_2 = nn.Parameter(
            a*torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, input_dim).

        Returns:
            Concatenated convolutions applied to alternating patch sizes.
            Output shape: (batch_size, out_channels, output_dim), where
            output_dim = input_dim // 5
        """

        # out_2 = (
        #    F.conv1d(x, self.filter_2, self.bias_2, stride=self.stride)
        # / (self.filter_2.size(1) * self.filter_2.size(2)) ** 0.5
        # )
        out_2 = F.conv1d(x, self.filter_2, self.bias_2, stride=self.stride)
        out_3 = F.conv1d(x[:, :, 2:], self.filter_3, self.bias_3, stride=self.stride)
        out = torch.cat((out_2, out_3), dim=-1)
        out = (
            out
            / (
                self.filter_2.size(1) * self.filter_2.size(2)
                + self.filter_3.size(1) * self.filter_3.size(2)
            )
            ** 0.5
        )
        return out


class MyConv1d_mixed_start_3(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include a bias term
        """
        super().__init__()
        self.filter_size_2 = 2
        self.filter_size_3 = 3
        self.stride = 5  # Stride should be the sum of both patch sizes (2+3)
        a=0.1
        # Two separate filters with proper initialization
        #filter_2 = torch.empty(out_channels, in_channels, self.filter_size_2)
        #torch.nn.init.kaiming_uniform_(
         #   filter_2, a=0.1
        #)  # Scaled-down initialization for filter 2
        #self.filter_2 = nn.Parameter(filter_2)

        self.filter_2=nn.Parameter(a*torch.randn(out_channels, in_channels, self.filter_size_2))

        #filter_3 = torch.empty(out_channels, in_channels, self.filter_size_3)
        #torch.nn.init.kaiming_uniform_(
         #   filter_3, a=1.0
        #)  # Standard initialization for filter 3
        #self.filter_3 = nn.Parameter(filter_3)
        self.filter_3=nn.Parameter(torch.randn(out_channels, in_channels, self.filter_size_3))
        # Two separate filters
        # self.filter_2 = nn.Parameter(
        #   torch.randn(out_channels, in_channels, self.filter_size_2)
        # )
        # self.filter_3 = nn.Parameter(
        #   torch.randn(out_channels, in_channels, self.filter_size_3)
        # )

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, input_dim).

        Returns:
            Concatenated convolutions applied to alternating patch sizes.
            Output shape: (batch_size, out_channels, output_dim), where
            output_dim = input_dim // 5
        """

        # Apply convolution separately on patch size 2
        # out_3 = (
        #   F.conv1d(x, self.filter_3, self.bias_3, stride=self.stride)
        #  / (self.filter_3.size(1) * self.filter_3.size(2)) ** 0.5
        # )
        out_3 = F.conv1d(x, self.filter_3, self.bias_3, stride=self.stride)
        # Apply convolution separately on patch size 3 (offset input by 2)
        out_2 = F.conv1d(x[:, :, 3:], self.filter_2, self.bias_2, stride=self.stride)
        # Concatenate along the last dimension
        out = torch.cat((out_3, out_2), dim=-1)
        out = (
            out
            / (
                self.filter_2.size(1) * self.filter_2.size(2)
                + self.filter_3.size(1) * self.filter_3.size(2)
            )
            ** 0.5
        )
        return out


class MyConv1d_2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include a bias term
        """
        super().__init__()
        self.filter_size_2 = 2

        self.stride = 1
        a=0.1
        # Two separate filters with proper initialization
        #filter_2 = torch.empty(out_channels, in_channels, self.filter_size_2)
        #torch.nn.init.kaiming_uniform_(
         #   filter_2, a=1.0
        #)  # Scaled-down initialization for filter 2
        #self.filter_2 = nn.Parameter(filter_2)

        self.filter_2=nn.Parameter(a*torch.randn(out_channels, in_channels, self.filter_size_2))
        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, input_dim).

        Returns:
            Concatenated convolutions applied to alternating patch sizes.
            Output shape: (batch_size, out_channels, output_dim), where
            output_dim = input_dim // 5
        """

        # Apply convolution separately on patch size 2
        # out_3 = (
        #   F.conv1d(x, self.filter_3, self.bias_3, stride=self.stride)
        #  / (self.filter_3.size(1) * self.filter_3.size(2)) ** 0.5
        # )
        out = (
            F.conv1d(x, self.filter_2, self.bias_2, stride=self.stride)
            / (self.filter_2.size(1) * self.filter_2.size(2)) ** 0.5
        )
        # Apply convolution separately on patch size 3 (offset input by 2)
        # Concatenate along the last dimension
        return out
    
class MyConv1d_3(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include a bias term
        """
        super().__init__()
        self.filter_size_3 = 3

        self.stride = 1

        # Two separate filters with proper initialization
        filter_3 = torch.empty(out_channels, in_channels, self.filter_size_3)
        torch.nn.init.kaiming_uniform_(
            filter_3, a=0.1
        )  # Scaled-down initialization for filter 2
        self.filter_3 = nn.Parameter(filter_3)

        # Bias terms
        if bias:
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_3", None)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, input_dim).

        Returns:
            Concatenated convolutions applied to alternating patch sizes.
            Output shape: (batch_size, out_channels, output_dim), where
            output_dim = input_dim // 5
        """

        # Apply convolution separately on patch size 2
        # out_3 = (
        #   F.conv1d(x, self.filter_3, self.bias_3, stride=self.stride)
        #  / (self.filter_3.size(1) * self.filter_3.size(2)) ** 0.5
        # )
        out = (
            F.conv1d(x, self.filter_3, self.bias_3, stride=self.stride)
            / (self.filter_3.size(1) * self.filter_3.size(2)) ** 0.5
        )
        # Apply convolution separately on patch size 3 (offset input by 2)
        # Concatenate along the last dimension
        return out


class MyLinear(nn.Module):

    def __init__(self, input_dim, out_dim, bias=False):
        """
        Args:
            input_dim: The input dimension.
            out_dim: The output dimension.
            bias: True for adding bias.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, input_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, *, input_dim).

        Returns:
            An affine transformation of x, tensor of size (batch_size, *, out_dim)
        """
        x = (
            F.linear(x, self.weight, self.bias) / (x.size(2) * x.size(1)) ** 0.5
        )  # standard scaling
        return x


class hCNN_mixed(nn.Module):
    def __init__(
        self,
        rule_sequence_type,
        in_channels,
        nn_dim,
        out_channels,
        num_layers,
        bias=False,
        norm="std",
    ):
        """
        Hierarchical CNN

        Args:
            input_dim: The input dimension.
            patch_size: The size of the patches.
            in_channels: The number of input channels.
            nn_dim: The number of hidden neurons per layer.
            out_channels: The output dimension.
            num_layers: The number of layers.
            bias: True for adding bias.
            norm: Scaling factor for the readout layer.
        """
        super().__init__()

        # receptive_field = patch_size**num_layers
        # assert input_dim % receptive_field == 0, 'patch_size**num_layers must divide input_dim!'
        if rule_sequence_type == 1:
            start_patches = [2, 3, 3, 2]
        elif rule_sequence_type == 2:
            start_patches = [3, 2, 2, 3]
        start_patches = start_patches[:num_layers]
        start_patches = start_patches[::-1]

        self.hidden = nn.Sequential(
            nn.Sequential(
                (
                    MyConv1d_mixed_start_2(in_channels, nn_dim, bias=bias)
                    if start_patches[0] == 2
                    else MyConv1d_mixed_start_3(in_channels, nn_dim, bias=bias)
                ),
                nn.ReLU(),
            ),
            *[
                nn.Sequential(
                    (
                        MyConv1d_mixed_start_2(nn_dim, nn_dim, bias=bias)
                        if start_patches[l] == 2
                        else MyConv1d_mixed_start_3(nn_dim, nn_dim, bias=bias)
                    ),
                    nn.ReLU(),
                )
                for l in range(1, num_layers - 1)
            ],
            (
                MyConv1d_2(nn_dim, nn_dim, bias=bias)
                if start_patches[-1] == 2
                else MyConv1d_3(nn_dim, nn_dim, bias=bias)
            ),
            nn.ReLU(),
        )

        self.readout = nn.Parameter(torch.randn(nn_dim, out_channels))
        if norm == "std":
            self.norm = nn_dim**0.5  # standard NTK scaling
        elif norm == "mf":
            self.norm = nn_dim  # mean-field scaling

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).

        Returns:
            Output of a hierarchical CNN, tensor of size (batch_size, out_dim)
        """
        x = self.hidden(x)
        x = x.mean(
            dim=[-1]
        )  # Global Average Pooling if the final spatial dimension is > 1
        x = x @ self.readout / self.norm
        return x


class MyConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, stride=1, bias=False):
        """
        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            filter_size: The size of the convolutional kernel
            stride: The stride (conv. ker. applied every stride pixels)
            bias: True for adding bias
        """
        super().__init__()

        self.filter_size = filter_size
        self.stride = stride
        self.filter = nn.Parameter(torch.randn(out_channels, in_channels, filter_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).

        Returns:
            The convolution of x with self.filter, tensor of size (batch_size, out_channels, out_dim),
            out_dim = (input_dim-filter_size)//stride+1
        """

        return (
            F.conv1d(x, self.filter, self.bias, stride=self.stride)
            / (self.filter.size(1) * self.filter.size(2)) ** 0.5
        )


class hCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        patch_size,
        in_channels,
        nn_dim,
        out_channels,
        num_layers,
        bias=False,
        norm="std",
    ):
        """
        Hierarchical CNN

        Args:
            input_dim: The input dimension.
            patch_size: The size of the patches.
            in_channels: The number of input channels.
            nn_dim: The number of hidden neurons per layer.
            out_channels: The output dimension.
            num_layers: The number of layers.
            bias: True for adding bias.
            norm: Scaling factor for the readout layer.
        """
        super().__init__()

        receptive_field = patch_size**num_layers
        assert (
            input_dim % receptive_field == 0
        ), "patch_size**num_layers must divide input_dim!"

        self.hidden = nn.Sequential(
            nn.Sequential(
                MyConv1d(in_channels, nn_dim, patch_size, stride=patch_size, bias=bias),
                nn.ReLU(),
            ),
            *[
                nn.Sequential(
                    MyConv1d(nn_dim, nn_dim, patch_size, stride=patch_size, bias=bias),
                    nn.ReLU(),
                )
                for l in range(1, num_layers)
            ],
        )
        self.readout = nn.Parameter(torch.randn(nn_dim, out_channels))
        if norm == "std":
            self.norm = nn_dim**0.5  # standard NTK scaling
        elif norm == "mf":
            self.norm = nn_dim  # mean-field scaling

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).

        Returns:
            Output of a hierarchical CNN, tensor of size (batch_size, out_dim)
        """
        x = self.hidden(x)
        x = x.mean(
            dim=[-1]
        )  # Global Average Pooling if the final spatial dimension is > 1
        x = x @ self.readout / self.norm
        return x
