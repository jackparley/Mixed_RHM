import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum



class MyConv1d_4(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include a bias term
        """
        super().__init__()
        self.filter_size_4 = 4

        self.stride = 1
        # a=0.1
        # Two separate filters with proper initialization
        # filter_2 = torch.empty(out_channels, in_channels, self.filter_size_2)
        # torch.nn.init.kaiming_uniform_(
        #   filter_2, a=1.0
        # )  # Scaled-down initialization for filter 2
        # self.filter_2 = nn.Parameter(filter_2)

        self.filter_4 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_4)
        )
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
            F.conv1d(x, self.filter_4, self.bias_2, stride=self.stride)
            / (self.filter_4.size(1) * self.filter_4.size(2)) ** 0.5
        )
        # Apply convolution separately on patch size 3 (offset input by 2)
        # Concatenate along the last dimension
        return out

class MyLoc1d_non_parallel(nn.Module):

    def __init__(
        self, input_dim, in_channels, out_channels, filter_size, stride=2, bias=False
    ):
        """
        Args:
            input_dim: The spatial dimension of the input
            in_channels: The number of input channels
            out_channels: The number of output channels
            filter_size: The size of the local kernel
            stride: The stride (kernel applied every stride pixels)
            bias: True for adding bias
        """
        super().__init__()

        self.input_dim = input_dim
        self.filter_size = filter_size
        self.stride = stride
        self.num_patches = (input_dim-filter_size)//stride + 1
        self.filter = nn.Parameter(
            torch.randn( out_channels, in_channels, self.num_patches, filter_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, self.num_patches))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).
        
        Returns:
            Tensor of size (batch_size, out_channels, num_patches).
        """
        batch_size, in_channels, _ = x.size()
        out = []

        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.filter_size
            patch = x[:, :, start:end]  # shape: (batch_size, in_channels, filter_size)
            filt = self.filter[:, :, i, :]  # shape: (out_channels, in_channels, filter_size)

            # Compute einsum per patch: [b, c, f] x [o, c, f] -> [b, o]
            prod = torch.einsum('bcf,ocf->bo', patch, filt)
            prod = prod * (in_channels * self.filter_size) ** -0.5

            if self.bias is not None:
                prod += self.bias[:, :, i]

            out.append(prod.unsqueeze(-1))  # shape: (batch_size, out_channels, 1)

        return torch.cat(out, dim=-1)  # shape: (batch_size, out_channels, num_patches)


class MyLoc1d(nn.Module):

    def __init__(
        self, input_dim, in_channels, out_channels, filter_size, stride=2, bias=False
    ):
        """
        Args:
            input_dim: The spatial dimension of the input
            in_channels: The number of input channels
            out_channels: The number of output channels
            filter_size: The size of the local kernel
            stride: The stride (kernel applied every stride pixels)
            bias: True for adding bias
        """
        super().__init__()

        self.input_dim = input_dim
        self.filter_size = filter_size
        self.stride = stride
        self.num_patches = (input_dim-filter_size)//stride + 1
        self.filter = nn.Parameter(
            torch.randn( out_channels, in_channels, self.num_patches, filter_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, self.num_patches))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).
        
        Returns:
            The convolution of x with self.filter, tensor of size (batch_size, out_channels, num_patches),
            num_patches = (input_dim-filter_size)//stride+1
        """
        out = F.unfold( # break input in patches [bs, in_channels, num_patches, filter_size]
            x.unsqueeze(-1),
            kernel_size=(self.filter_size,1),
            dilation=1,
            padding=0,
            stride=self.stride,
        ).reshape(-1, self.filter.size(1), self.filter.size(3), self.filter.size(2)).transpose(-1,-2)

        out = out[:, None] * self.filter # [bs, out_channels, in_channels, num_patches, filter_size]
        out = out.sum(dim=[-1,-3]) *(self.filter.size(1)*self.filter.size(3))**-.5  # [bs, out_channels, num_patches]
        if self.bias is not None:
            out += self.bias

        return out



class hCNN_no_sharing_Gen(nn.Module):
    def __init__(
        self,
        in_channels,
        nn_dim,
        out_channels,
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
        input_dim=10
        filter_size=4

        self.conv1 = MyLoc1d(input_dim,in_channels, nn_dim,filter_size,stride=2, bias=bias)
        self.relu1 = nn.ReLU()
        self.conv2 = MyConv1d_4(nn_dim, nn_dim, bias=bias)
        self.relu2 = nn.ReLU()

        #self.hidden = nn.Sequential(
         #       (
          #          MyConv1d_no_sharing(in_channels, nn_dim, bias=bias)
                    
           #     ),
            #    nn.ReLU(),
             #       (
              #      MyConv1d_2(nn_dim, nn_dim_2, bias=bias)
                    
               # ),
                #nn.ReLU(),
            #)

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
        x = self.conv1(x)
        #print(x.shape)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        #print(x.shape)
        x = x.squeeze(-1)
        #x = self.hidden(x)
        x = x @ self.readout / self.norm
        return x
    







class MyConv1d_no_sharing(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include a bias term
        """
        super().__init__()
        self.filter_size_3_1 = 3
        self.filter_size_3_2 = 3
        self.stride = 10  # Stride should be the sum of both patch sizes (2+3)

        # Two separate filters
        self.filter_3_1 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3_1)
        )
        self.filter_3_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3_2)
        )

        # Bias terms
        if bias:
            self.bias_3_1 = nn.Parameter(torch.randn(out_channels))
            self.bias_3_2 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_3_1", None)
            self.register_parameter("bias_3_2", None)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, in_channels, input_dim).

        Returns:
            Interleaved convolutions applied to alternating patch sizes.
            Output shape: (batch_size, out_channels, output_dim), where
            output_dim = (num valid patches from both filters, interleaved).
        """

        # Apply convolution separately on patch size 2
        out_3_1 = (
            F.conv1d(x, self.filter_3_1, self.bias_3_1, stride=self.stride)
            / (self.filter_3_1.size(1) * self.filter_3_1.size(2)) ** 0.5
        )

        # Apply convolution separately on patch size 3 (offset input by 2)
        out_3_2 = (
            F.conv1d(x[:, :, 2:], self.filter_3_2, self.bias_3_2, stride=self.stride)
            / (self.filter_3_2.size(1) * self.filter_3_2.size(2)) ** 0.5
        )
        out = torch.cat((out_3_1, out_3_2), dim=-1)


        return out



class MyConv1d_mixed_interleave_start_2(nn.Module):
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

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
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
            Interleaved convolutions applied to alternating patch sizes.
            Output shape: (batch_size, out_channels, output_dim), where
            output_dim = (num valid patches from both filters, interleaved).
        """

        # Apply convolution separately on patch size 2
        out_2 = (
            F.conv1d(x, self.filter_2, self.bias_2, stride=self.stride)
            / (self.filter_2.size(1) * self.filter_2.size(2)) ** 0.5
        )

        # Apply convolution separately on patch size 3 (offset input by 2)
        out_3 = (
            F.conv1d(x[:, :, 2:], self.filter_3, self.bias_3, stride=self.stride)
            / (self.filter_3.size(1) * self.filter_3.size(2)) ** 0.5
        )

        # Determine min valid length to interleave
        min_length = min(out_2.shape[-1], out_3.shape[-1])

        # Initialize interleaved output tensor
        batch_size, out_channels = out_2.shape[:2]
        total_length = out_2.shape[-1] + out_3.shape[-1]
        interleaved_out = torch.zeros(
            batch_size, out_channels, total_length, device=x.device
        )

        # Interleave up to min_length
        interleaved_out[:, :, 0 : 2 * min_length : 2] = out_3[
            :, :, :min_length
        ]  # Place out_3 at even indices
        interleaved_out[:, :, 1 : 2 * min_length : 2] = out_2[
            :, :, :min_length
        ]  # Place out_2 at odd indices

        # Append remaining values from the longer output (if any)
        if out_3.shape[-1] > min_length:
            interleaved_out[:, :, 2 * min_length :] = out_3[:, :, min_length:]
        elif out_2.shape[-1] > min_length:
            interleaved_out[:, :, 2 * min_length :] = out_2[:, :, min_length:]

        return interleaved_out


class MyConv1d_mixed_interleave_start_3(nn.Module):
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

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
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
            Interleaved convolutions applied to alternating patch sizes,
            starting with a patch of length 3.
        """

        # Apply convolution separately on patch size 3 (start with this one)
        out_3 = (
            F.conv1d(x, self.filter_3, self.bias_3, stride=self.stride)
            / (self.filter_3.size(1) * self.filter_3.size(2)) ** 0.5
        )

        # Apply convolution separately on patch size 2 (offset input by 3 instead of 2)
        out_2 = (
            F.conv1d(x[:, :, 3:], self.filter_2, self.bias_2, stride=self.stride)
            / (self.filter_2.size(1) * self.filter_2.size(2)) ** 0.5
        )

        # Determine min valid length to interleave
        min_length = min(out_2.shape[-1], out_3.shape[-1])

        # Initialize interleaved output tensor
        batch_size, out_channels = out_2.shape[:2]
        total_length = out_2.shape[-1] + out_3.shape[-1]
        interleaved_out = torch.zeros(
            batch_size, out_channels, total_length, device=x.device
        )

        # Interleave up to min_length
        interleaved_out[:, :, 0 : 2 * min_length : 2] = out_3[
            :, :, :min_length
        ]  # Place out_3 at even indices
        interleaved_out[:, :, 1 : 2 * min_length : 2] = out_2[
            :, :, :min_length
        ]  # Place out_2 at odd indices

        # Append remaining values from the longer output (if any)
        if out_3.shape[-1] > min_length:
            interleaved_out[:, :, 2 * min_length :] = out_3[:, :, min_length:]
        elif out_2.shape[-1] > min_length:
            interleaved_out[:, :, 2 * min_length :] = out_2[:, :, min_length:]

        return interleaved_out


class MyConv1d_mixed_start_2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include a bias term
        """
        super().__init__()
        a = 0.1
        self.filter_size_2 = 2
        self.filter_size_3 = 3
        self.stride = 5  # Stride should be the sum of both patch sizes (2+3)

        # Two separate filters
        self.filter_2 = nn.Parameter(
            a * torch.randn(out_channels, in_channels, self.filter_size_2)
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
        a = 0.1
        # Two separate filters with proper initialization
        # filter_2 = torch.empty(out_channels, in_channels, self.filter_size_2)
        # torch.nn.init.kaiming_uniform_(
        #   filter_2, a=0.1
        # )  # Scaled-down initialization for filter 2
        # self.filter_2 = nn.Parameter(filter_2)

        self.filter_2 = nn.Parameter(
            a * torch.randn(out_channels, in_channels, self.filter_size_2)
        )

        # filter_3 = torch.empty(out_channels, in_channels, self.filter_size_3)
        # torch.nn.init.kaiming_uniform_(
        #   filter_3, a=1.0
        # )  # Standard initialization for filter 3
        # self.filter_3 = nn.Parameter(filter_3)
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )
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
        # a=0.1
        # Two separate filters with proper initialization
        # filter_2 = torch.empty(out_channels, in_channels, self.filter_size_2)
        # torch.nn.init.kaiming_uniform_(
        #   filter_2, a=1.0
        # )  # Scaled-down initialization for filter 2
        # self.filter_2 = nn.Parameter(filter_2)

        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
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
        # a=0.1
        # Two separate filters with proper initialization
        # filter_2 = torch.empty(out_channels, in_channels, self.filter_size_2)
        # torch.nn.init.kaiming_uniform_(
        #   filter_2, a=1.0
        # )  # Scaled-down initialization for filter 2
        # self.filter_2 = nn.Parameter(filter_2)

        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )
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


class MyConv1d_3_stride_2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include a bias term
        """
        super().__init__()
        self.filter_size_3 = 3

        self.stride = 2
        # a=0.1
        # Two separate filters with proper initialization
        # filter_2 = torch.empty(out_channels, in_channels, self.filter_size_2)
        # torch.nn.init.kaiming_uniform_(
        #   filter_2, a=1.0
        # )  # Scaled-down initialization for filter 2
        # self.filter_2 = nn.Parameter(filter_2)

        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )
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

class MyConv1d_3_stride_3(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include a bias term
        """
        super().__init__()
        self.filter_size_3 = 3

        self.stride = 3
        # a=0.1
        # Two separate filters with proper initialization
        # filter_2 = torch.empty(out_channels, in_channels, self.filter_size_2)
        # torch.nn.init.kaiming_uniform_(
        #   filter_2, a=1.0
        # )  # Scaled-down initialization for filter 2
        # self.filter_2 = nn.Parameter(filter_2)

        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )
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
                    MyConv1d_mixed_interleave_start_2(in_channels, nn_dim, bias=bias)
                    if start_patches[0] == 2
                    else MyConv1d_mixed_interleave_start_3(
                        in_channels, nn_dim, bias=bias
                    )
                ),
                nn.ReLU(),
            ),
            *[
                nn.Sequential(
                    (
                        MyConv1d_mixed_interleave_start_2(nn_dim, nn_dim, bias=bias)
                        if start_patches[l] == 2
                        else MyConv1d_mixed_interleave_start_3(
                            nn_dim, nn_dim, bias=bias
                        )
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



class hCNN_sharing(nn.Module):
    def __init__(
        self,
        in_channels,
        nn_dim,
        nn_dim_2,
        out_channels,
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

        self.hidden = nn.Sequential(
                (
                    MyConv1d_3_stride_2(in_channels, nn_dim, bias=bias)
                    
                ),
                nn.ReLU(),
                    (
                    MyConv1d_2(nn_dim, nn_dim_2, bias=bias)
                    
                ),
                nn.ReLU(),
            )

        self.readout = nn.Parameter(torch.randn(nn_dim_2, out_channels))
        if norm == "std":
            self.norm = nn_dim_2**0.5  # standard NTK scaling
        elif norm == "mf":
            self.norm = nn_dim_2  # mean-field scaling

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).

        Returns:
            Output of a hierarchical CNN, tensor of size (batch_size, out_dim)
        """
        x = self.hidden(x)
        x = x.squeeze(-1)
        x = x @ self.readout / self.norm
        return x


class hCNN_no_sharing(nn.Module):
    def __init__(
        self,
        in_channels,
        nn_dim,
        nn_dim_2,
        out_channels,
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

        self.conv1 = MyConv1d_no_sharing(in_channels, nn_dim, bias=bias)
        self.relu1 = nn.ReLU()
        self.conv2 = MyConv1d_2(nn_dim, nn_dim_2, bias=bias)
        self.relu2 = nn.ReLU()

        #self.hidden = nn.Sequential(
         #       (
          #          MyConv1d_no_sharing(in_channels, nn_dim, bias=bias)
                    
           #     ),
            #    nn.ReLU(),
             #       (
              #      MyConv1d_2(nn_dim, nn_dim_2, bias=bias)
                    
               # ),
                #nn.ReLU(),
            #)

        self.readout = nn.Parameter(torch.randn(nn_dim_2, out_channels))
        if norm == "std":
            self.norm = nn_dim_2**0.5  # standard NTK scaling
        elif norm == "mf":
            self.norm = nn_dim_2  # mean-field scaling

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).

        Returns:
            Output of a hierarchical CNN, tensor of size (batch_size, out_dim)
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        #print(x.shape)
        x = x.squeeze(-1)
        #x = self.hidden(x)
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


class MyConv1d_top_fix(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        """
        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            bias: True for adding bias
        """
        super().__init__()

        self.filter_size = 3
        self.stride=5
        self.filter = nn.Parameter(torch.randn(out_channels, in_channels, self.filter_size))
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
            out_dim = input_dim
        """
        out_1=F.conv1d(x[:,:,:3], self.filter, self.bias, stride=self.stride)/ (self.filter.size(1) * self.filter.size(2)) ** 0.5
        out_2=F.conv1d(torch.cat((x[:, :, 0:2], x[:, :, 3:4]), dim=2), self.filter, self.bias, stride=self.stride)/ (self.filter.size(1) * self.filter.size(2)) ** 0.5
        out=(out_1 + out_2)/2
        return out


class hCNN_Gen_top_fix(nn.Module):
    def __init__(
        self,
        in_channels,
        nn_dim,
        out_channels,
        num_layers,
        final_dim,    # This is the input size after convolutions
        bias=False,
        norm="std",
    ):
        """
        Generic Hierarchical CNN supporting predefined final output lengths.

        Args:
            in_channels: The number of input channels.
            nn_dim: The number of hidden neurons per layer.
            out_channels: The output dimension.
            num_layers: The number of convolutional layers.
            final_dim: The spatial size (length) after the convolution layers.
            bias: True for adding bias.
            norm: Scaling factor for the readout layer.
        """
        super().__init__()
        filter_size = 4
        stride_length = 2

        self.hidden = nn.Sequential(
                MyConv1d(in_channels, nn_dim, filter_size, stride=stride_length, bias=bias),
                nn.ReLU(),
                MyConv1d_top_fix(nn_dim, nn_dim, bias=bias),
                    nn.ReLU(),
                )
        self.nn_dim = nn_dim  # Number of hidden neurons per layer
        self.final_dim = final_dim  # Predefined spatial size after conv layers

        # Define readout layer with flattened input
        self.readout = nn.Parameter(torch.randn(nn_dim * final_dim, out_channels))

        # Norm scaling for feature learning regime
        if norm == "std":
            self.norm = (nn_dim * final_dim) ** 0.5  # standard NTK scaling
        elif norm == "mf":
            self.norm = nn_dim * final_dim           # mean-field scaling

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).

        Returns:
            Output of a hierarchical CNN, tensor of size (batch_size, out_dim)
        """
        x = self.hidden(x)  # (batch_size, nn_dim, final_dim)
        batch_size = x.shape[0]

        # Flatten the conv output along feature dimensions
        x = x.reshape(batch_size, self.nn_dim * self.final_dim)

        # Linear readout with mean-field scaling
        x = x @ self.readout / self.norm

        return x




class hCNN_Gen(nn.Module):
    def __init__(
        self,
        in_channels,
        nn_dim,
        out_channels,
        num_layers,
        final_dim,    # This is the input size after convolutions
        bias=False,
        norm="std",
    ):
        """
        Generic Hierarchical CNN supporting predefined final output lengths.

        Args:
            in_channels: The number of input channels.
            nn_dim: The number of hidden neurons per layer.
            out_channels: The output dimension.
            num_layers: The number of convolutional layers.
            final_dim: The spatial size (length) after the convolution layers.
            bias: True for adding bias.
            norm: Scaling factor for the readout layer.
        """
        super().__init__()
        filter_size = 4
        stride_length = 2

        self.hidden = nn.Sequential(
            nn.Sequential(
                MyConv1d(in_channels, nn_dim, filter_size, stride=stride_length, bias=bias),
                nn.ReLU(),
            ),
            *[
                nn.Sequential(
                    MyConv1d(nn_dim, nn_dim, filter_size, stride=stride_length, bias=bias),
                    nn.ReLU(),
                )
                for _ in range(1, num_layers)
            ],
        )
        self.nn_dim = nn_dim  # Number of hidden neurons per layer
        self.final_dim = final_dim  # Predefined spatial size after conv layers

        # Define readout layer with flattened input
        self.readout = nn.Parameter(torch.randn(nn_dim * final_dim, out_channels))

        # Norm scaling for feature learning regime
        if norm == "std":
            self.norm = (nn_dim * final_dim) ** 0.5  # standard NTK scaling
        elif norm == "mf":
            self.norm = nn_dim * final_dim           # mean-field scaling

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, in_channels, input_dim).

        Returns:
            Output of a hierarchical CNN, tensor of size (batch_size, out_dim)
        """
        x = self.hidden(x)  # (batch_size, nn_dim, final_dim)
        batch_size = x.shape[0]

        # Flatten the conv output along feature dimensions
        x = x.reshape(batch_size, self.nn_dim * self.final_dim)

        # Linear readout with mean-field scaling
        x = x @ self.readout / self.norm

        return x




class hCNN_Gen_MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        nn_dim,
        out_channels,
        num_layers,
        final_dim,
        mlp_dim,      # New argument for hidden dimension in MLP
        bias=False,
        norm="std",
    ):
        super().__init__()
        filter_size = 4
        stride_length = 2

        self.hidden = nn.Sequential(
            nn.Sequential(
                MyConv1d(in_channels, nn_dim, filter_size, stride=stride_length, bias=bias),
                nn.ReLU(),
            ),
            *[
                nn.Sequential(
                    MyConv1d(nn_dim, nn_dim, filter_size, stride=stride_length, bias=bias),
                    nn.ReLU(),
                )
                for _ in range(1, num_layers)
            ],
        )

        self.nn_dim = nn_dim
        self.final_dim = final_dim
        flattened_dim = nn_dim * final_dim

        # New MLP layer before readout
        self.mlp = nn.Sequential(
            nn.Linear(flattened_dim, mlp_dim),
            nn.ReLU()
        )

        # Readout layer
        self.readout = nn.Parameter(torch.randn(mlp_dim, out_channels))

        if norm == "std":
            self.norm = mlp_dim ** 0.5
        elif norm == "mf":
            self.norm = mlp_dim

    def forward(self, x):
        x = self.hidden(x)  # (batch_size, nn_dim, final_dim)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.nn_dim * self.final_dim)
        x = self.mlp(x)  # Pass through MLP layer
        x = x @ self.readout / self.norm
        return x





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


class ComputeSumOfSquares(nn.Module):
    """Custom module to compute sum of squares and pass both input and sum forward."""

    def forward(self, x):
        sum_of_squares = torch.sum(x**2, dim=(1, 2), keepdim=True)
        sum_of_squares = torch.round(sum_of_squares).to(torch.int)
        sum_of_squares = sum_of_squares.squeeze()

        return x, sum_of_squares  # Returning a tuple


class TensorChopping(nn.Module):
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_features, input_length + 1)
        
        Returns:
            original_x: Tensor of shape (batch_size, num_features, input_length)
            tree_topologies: Tensor of shape (batch_size,)
        """
        # Separate the last slice
        original_x = x[:, :, :-1]  # Drop last column
        tree_topologies = x[:, 0, -1].long()  # Take topology info from the first feature
        return original_x, tree_topologies


class TupleReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x, sum_of_squares = inputs  # Unpack tuple
        return self.relu(x), sum_of_squares  # Apply ReLU only to x


class MyConv1d_ell_1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.filter_size_2 = 2
        self.filter_size_3 = 3
        self.stride = 1  # Stride should be the sum of both patch sizes (2+3)

        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, inputs):
        x, sum_of_squares = inputs  # Unpack tuple

        out_2 = (
            F.conv1d(x, self.filter_2, self.bias_2, stride=self.stride)
            / (self.filter_2.size(1) * self.filter_2.size(2)) ** 0.5
        )
        out_3 = (
            F.conv1d(x, self.filter_3, self.bias_3, stride=self.stride)
            / (self.filter_3.size(1) * self.filter_3.size(2)) ** 0.5
        )

        pad_2 = torch.zeros(
            out_2.shape[0], out_2.shape[1], 1, device=out_2.device, dtype=out_2.dtype
        )
        out_2_padded = torch.cat([out_2, pad_2], dim=2)
        pad_3 = torch.zeros(
            out_3.shape[0], out_3.shape[1], 2, device=out_3.device, dtype=out_3.dtype
        )
        out_3_padded = torch.cat([out_3, pad_3], dim=2)

        interleaved_out = torch.stack((out_2_padded, out_3_padded), dim=3).view(
            out_2.shape[0], out_2.shape[1], -1
        )

        return interleaved_out, sum_of_squares  # Return tuple


def nested_ranges_as_tensor(i, j, min_split):
    # Generate pairs (k1, k2)
    pairs = [
        (k1, k2)
        for k1 in range(i + min_split, j - min_split + 1)
        for k2 in range(k1 + 1, j)
    ]

    # Convert to a tensor
    if pairs:
        return torch.tensor(pairs,dtype=torch.int)  # Shape: [num_pairs, 2]
    else:
        return torch.empty(
            (0, 2), dtype=torch.int
        )  # Handle case where no pairs exist

def manual_conv1d_stacked(input_tensor, filters, stride):
    """
    Explicit implementation of 1D convolution with stacked filters.
    
    Args:
    - input_tensor: (batch_size, in_channels, width) input signal
    - filters: (num_filters, out_channels, in_channels, kernel_size) stacked convolution filters
    - bias: (out_channels,) optional bias
    - stride: Stride for convolution
    - padding: Zero padding
    
    Returns:
    - output: Convolved tensor of shape (batch_size, out_channels, output_width)
    """
    #batch_size, in_channels, width = input_tensor.shape
    num_filters, out_channels, _, kernel_size = filters.shape  # Filters now have an extra `num_filters` dim


    # Compute correct output width
    #output_width = (width - kernel_size) // stride + 1

    # Extract sliding windows using unfold
    unfolded = input_tensor.unfold(dimension=2, size=kernel_size, step=stride)  # Shape: (batch_size, in_channels, output_width, kernel_size)

    # Reshape for einsum
    unfolded = unfolded.permute(0, 2, 1, 3)  # Shape: (batch_size, output_width, in_channels, kernel_size)

    # Perform convolution with stacked filters
    output = einsum('bwik,foik->bfow', unfolded, filters)  # (batch_size, num_filters, out_channels, output_width)

    # Sum across the `num_filters` dimension
    output = output.sum(dim=1)  # (batch_size, out_channels, output_width)
    return output


class MyConv1d_ell_3_last(nn.Module):
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

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        l = 3
        self.n = 27
        self.min_d = 2**l
        self.max_d = 3 ** (l)
        self.min_split = 2 ** (l - 1)
        self.max_split = 3 ** (l - 1)
        self.n_span = 3 ** (l - 1) - 2 ** (l - 1) + 1
        # Dictionary to store kk_0 lists
        self.stride = self.n_span*self.n

        self.kk_0_dict = {}
        self.pairs_0_dict = {}

        for length in range(self.min_d, self.max_d + 1):  # Length of the span
            kk_0 = np.array(range(self.min_split, length - self.min_split + 1))
            kk_0 = kk_0[(kk_0 <= self.max_split) & (kk_0 >= (length - self.max_split))]
            self.kk_0_dict[length] = kk_0.tolist()
            pairs_0 = nested_ranges_as_tensor(0, length, self.min_split)
            condition_1 = pairs_0[:, 0] <= self.max_split
            condition_2 = ((pairs_0[:, 1] - pairs_0[:, 0]) <= self.max_split) & (
                (pairs_0[:, 1] - pairs_0[:, 0]) >= self.min_split
            )
            condition_3 = (pairs_0[:, 1] >= (length - self.max_split)) & (
                length - pairs_0[:, 1] >= self.min_split
            )
            # Combine all conditions
            pairs_0 = pairs_0[condition_1 & condition_2 & condition_3]
            self.pairs_0_dict[length] = pairs_0

        # Pruning condition: keep elements <= max_split and >= (length - max_split)

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, inputs):
        x, sum_of_squares = inputs  # Unpack tuple
        # Apply convolution separately on patch size 2
        outs = []
        for length in range(self.min_d, self.max_d + 1):
            kk_0 = self.kk_0_dict[length]
            # print(kk_0)
            kk_0 = torch.tensor(kk_0, dtype=torch.int)  # Shape: (num_kk,)
            # print(out_d.shape)
            dp_1 = kk_0  # Shape: (num_kk,)
            dp_2 = length - kk_0  # Shape: (num_kk,)
            full_filter = torch.zeros(
                self.filter_2.shape[0],  # Out channels
                self.filter_2.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_2.device,
            )
            #out_d = torch.zeros(
             #   x.shape[0], self.filter_2.shape[0], self.n - length + 1, device=x.device
            #)

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0
            indices = torch.stack(
                [dp_1 - self.min_split, self.n_span * dp_1 + dp_2 - self.min_split], dim=1
            ).permute(1,0)  # Shape: (2,num_kk)

            num_filters_2 = indices.shape[1]

            # Replicate full_filter across a new dimension for the filters.
            stacked_filters_2 = full_filter.unsqueeze(-1).expand(-1, -1, -1, num_filters_2).clone()
            # Create filter indices for the last dimension.
            filter_indices = torch.arange(num_filters_2)

            # Expand filter_2 slices to shape [7,6,num_filters]
            filter_2_slice0 = self.filter_2[:, :, 0].unsqueeze(-1).expand(self.filter_2.size(0), self.filter_2.size(1), num_filters_2)
            filter_2_slice1 = self.filter_2[:, :, 1].unsqueeze(-1).expand(self.filter_2.size(0), self.filter_2.size(1), num_filters_2)

            # Update the positions specified by indices for each filter.
            stacked_filters_2[:, :, indices[0, :], filter_indices] = filter_2_slice0/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )
            stacked_filters_2[:, :, indices[1, :], filter_indices] = filter_2_slice1/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )

            # Reshape the stacked filters to shape [num_filters_2, 7, 6, 10]

            stacked_filters_2=stacked_filters_2.permute(3,0,1,2)


            #out_d+=manual_conv1d_stacked(x, stacked_filters, stride=self.stride)/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )
            
            #Ternary filters

            pairs_0 = self.pairs_0_dict[length]
            #pairs_0 = torch.tensor(pairs_0, dtype=torch.int)  # Shape: (num_kk,2)
            full_filter = torch.zeros(
                self.filter_3.shape[0],  # Out channels
                self.filter_3.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_3.device,
            )

            dp_1 = pairs_0[:, 0]
            dp_2 = pairs_0[:, 1] - pairs_0[:, 0]
            dp_3 = length - pairs_0[:, 1]

            indices = torch.stack(
                [
                    dp_1 - self.min_split,
                    self.n_span * dp_1 + dp_2 - self.min_split,
                    self.n_span * (dp_1 + dp_2) + dp_3 - self.min_split,
                ],
                dim=1,
            ).permute(
                1, 0
            )  # Shape: (3,num_kk)

            num_filters_3 = indices.shape[1]

            # Replicate full_filter across a new dimension for the filters.
            stacked_filters_3 = full_filter.unsqueeze(-1).expand(-1, -1, -1, num_filters_3).clone()
            # Create filter indices for the last dimension.
            filter_indices = torch.arange(num_filters_3)

            # Expand filter_2 slices to shape [7,6,num_filters]
            filter_3_slice0 = self.filter_3[:, :, 0].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)
            filter_3_slice1 = self.filter_3[:, :, 1].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)
            filter_3_slice2 = self.filter_3[:, :, 2].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)

            # Update the positions specified by indices for each filter.
            stacked_filters_3[:, :, indices[0, :], filter_indices] = filter_3_slice0/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )
            stacked_filters_3[:, :, indices[1, :], filter_indices] = filter_3_slice1/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )
            stacked_filters_3[:, :, indices[2, :], filter_indices] = filter_3_slice2/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )

            # Reshape the stacked filters to shape [num_filters, 7, 6, 10]

            stacked_filters_3=stacked_filters_3.permute(3,0,1,2) # Shape: (num_filters_3, out_channels, in_channels, kernel_size)
            
            stacked_filters = torch.cat((stacked_filters_2, stacked_filters_3), dim=0)
            
            out_d=manual_conv1d_stacked(x, stacked_filters, stride=self.stride)

            #pad = torch.zeros(
             #   out_d.shape[0],
              #  out_d.shape[1],
               # self.n - out_d.shape[2],
                #device=out_d.device,
                #dtype=out_d.dtype,
            #)
            #out_d_padded = torch.cat([out_d, pad], dim=2)
            outs.append(out_d)
        # Step 1: Stack all elements in outs along a new dimension (dim=3)
        intermediate = torch.stack(
            outs, dim=3
        )  # Shape: (batch_size, out_channels, seq_len, num_spans)

        # Step 2: Reshape to interleave
        interleaved_out = intermediate.view(
            intermediate.shape[0], intermediate.shape[1], -1
        )

        return interleaved_out, sum_of_squares  # Return tuple




class MyConv1d_ell_2_last(nn.Module):
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

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        l = 2
        self.n = 9
        self.min_d = 2**l
        self.max_d = 3 ** (l)
        self.min_split = 2 ** (l - 1)
        self.max_split = 3 ** (l - 1)
        self.n_span = 3 ** (l - 1) - 2 ** (l - 1) + 1
        # Dictionary to store kk_0 lists
        self.stride = self.n_span*self.n

        self.kk_0_dict = {}
        self.pairs_0_dict = {}

        for length in range(self.min_d, self.max_d + 1):  # Length of the span
            kk_0 = np.array(range(self.min_split, length - self.min_split + 1))
            kk_0 = kk_0[(kk_0 <= self.max_split) & (kk_0 >= (length - self.max_split))]
            self.kk_0_dict[length] = kk_0.tolist()
            pairs_0 = nested_ranges_as_tensor(0, length, self.min_split)
            condition_1 = pairs_0[:, 0] <= self.max_split
            condition_2 = ((pairs_0[:, 1] - pairs_0[:, 0]) <= self.max_split) & (
                (pairs_0[:, 1] - pairs_0[:, 0]) >= self.min_split
            )
            condition_3 = (pairs_0[:, 1] >= (length - self.max_split)) & (
                length - pairs_0[:, 1] >= self.min_split
            )
            # Combine all conditions
            pairs_0 = pairs_0[condition_1 & condition_2 & condition_3]
            self.pairs_0_dict[length] = pairs_0

        # Pruning condition: keep elements <= max_split and >= (length - max_split)

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, inputs):
        x, sum_of_squares = inputs  # Unpack tuple
        # Apply convolution separately on patch size 2
        outs = []
        for length in range(self.min_d, self.max_d + 1):
            kk_0 = self.kk_0_dict[length]
            # print(kk_0)
            kk_0 = torch.tensor(kk_0, dtype=torch.int)  # Shape: (num_kk,)
            # print(out_d.shape)
            dp_1 = kk_0  # Shape: (num_kk,)
            dp_2 = length - kk_0  # Shape: (num_kk,)
            full_filter = torch.zeros(
                self.filter_2.shape[0],  # Out channels
                self.filter_2.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_2.device,
            )
            #out_d = torch.zeros(
             #   x.shape[0], self.filter_2.shape[0], self.n - length + 1, device=x.device
            #)

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0
            indices = torch.stack(
                [dp_1 - self.min_split, self.n_span * dp_1 + dp_2 - self.min_split], dim=1
            ).permute(1,0)  # Shape: (2,num_kk)

            num_filters_2 = indices.shape[1]

            # Replicate full_filter across a new dimension for the filters.
            stacked_filters_2 = full_filter.unsqueeze(-1).expand(-1, -1, -1, num_filters_2).clone()
            # Create filter indices for the last dimension.
            filter_indices = torch.arange(num_filters_2)

            # Expand filter_2 slices to shape [7,6,num_filters]
            filter_2_slice0 = self.filter_2[:, :, 0].unsqueeze(-1).expand(self.filter_2.size(0), self.filter_2.size(1), num_filters_2)
            filter_2_slice1 = self.filter_2[:, :, 1].unsqueeze(-1).expand(self.filter_2.size(0), self.filter_2.size(1), num_filters_2)

            # Update the positions specified by indices for each filter.
            stacked_filters_2[:, :, indices[0, :], filter_indices] = filter_2_slice0/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )
            stacked_filters_2[:, :, indices[1, :], filter_indices] = filter_2_slice1/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )

            # Reshape the stacked filters to shape [num_filters_2, 7, 6, 10]

            stacked_filters_2=stacked_filters_2.permute(3,0,1,2)


            #out_d+=manual_conv1d_stacked(x, stacked_filters, stride=self.stride)/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )
            
            #Ternary filters

            pairs_0 = self.pairs_0_dict[length]
            #pairs_0 = torch.tensor(pairs_0, dtype=torch.int)  # Shape: (num_kk,2)
            full_filter = torch.zeros(
                self.filter_3.shape[0],  # Out channels
                self.filter_3.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_3.device,
            )

            dp_1 = pairs_0[:, 0]
            dp_2 = pairs_0[:, 1] - pairs_0[:, 0]
            dp_3 = length - pairs_0[:, 1]

            indices = torch.stack(
                [
                    dp_1 - self.min_split,
                    self.n_span * dp_1 + dp_2 - self.min_split,
                    self.n_span * (dp_1 + dp_2) + dp_3 - self.min_split,
                ],
                dim=1,
            ).permute(
                1, 0
            )  # Shape: (3,num_kk)

            num_filters_3 = indices.shape[1]

            # Replicate full_filter across a new dimension for the filters.
            stacked_filters_3 = full_filter.unsqueeze(-1).expand(-1, -1, -1, num_filters_3).clone()
            # Create filter indices for the last dimension.
            filter_indices = torch.arange(num_filters_3)

            # Expand filter_2 slices to shape [7,6,num_filters]
            filter_3_slice0 = self.filter_3[:, :, 0].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)
            filter_3_slice1 = self.filter_3[:, :, 1].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)
            filter_3_slice2 = self.filter_3[:, :, 2].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)

            # Update the positions specified by indices for each filter.
            stacked_filters_3[:, :, indices[0, :], filter_indices] = filter_3_slice0/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )
            stacked_filters_3[:, :, indices[1, :], filter_indices] = filter_3_slice1/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )
            stacked_filters_3[:, :, indices[2, :], filter_indices] = filter_3_slice2/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )

            # Reshape the stacked filters to shape [num_filters, 7, 6, 10]

            stacked_filters_3=stacked_filters_3.permute(3,0,1,2) # Shape: (num_filters_3, out_channels, in_channels, kernel_size)
            
            stacked_filters = torch.cat((stacked_filters_2, stacked_filters_3), dim=0)
            
            out_d=manual_conv1d_stacked(x, stacked_filters, stride=self.stride)

            #pad = torch.zeros(
             #   out_d.shape[0],
              #  out_d.shape[1],
               # self.n - out_d.shape[2],
                #device=out_d.device,
                #dtype=out_d.dtype,
            #)
            #out_d_padded = torch.cat([out_d, pad], dim=2)
            outs.append(out_d)
        # Step 1: Stack all elements in outs along a new dimension (dim=3)
        intermediate = torch.stack(
            outs, dim=3
        )  # Shape: (batch_size, out_channels, seq_len, num_spans)

        # Step 2: Reshape to interleave
        interleaved_out = intermediate.view(
            intermediate.shape[0], intermediate.shape[1], -1
        )

        return interleaved_out, sum_of_squares  # Return tuple




class MyConv1d_ell_2_last_tree_topologies(nn.Module):
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

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        l = 2
        self.n = 9
        self.min_d = 2**l
        self.max_d = 3 ** (l)
        self.min_split = 2 ** (l - 1)
        self.max_split = 3 ** (l - 1)
        self.n_span = 3 ** (l - 1) - 2 ** (l - 1) + 1
        # Dictionary to store kk_0 lists
        self.stride = self.n_span*self.n

        self.kk_0_dict = {}
        self.pairs_0_dict = {}


        self.kk_0_dict[0]=[2]
        self.kk_0_dict[1]=[2]
        self.kk_0_dict[2]=[3]
        self.kk_0_dict[3]=[3]
        self.kk_0_dict[4]=[]
        self.kk_0_dict[5]=[]
        self.kk_0_dict[6]=[]
        self.kk_0_dict[7]=[]  
        self.kk_0_dict[8]=[]
        self.kk_0_dict[9]=[]
        self.kk_0_dict[10]=[]
        self.kk_0_dict[11]=[]

        self.pairs_0_dict[0]=torch.empty((0, 2), dtype=torch.int32)
        self.pairs_0_dict[1]=torch.empty((0, 2), dtype=torch.int32)
        self.pairs_0_dict[2]=torch.empty((0, 2), dtype=torch.int32)
        self.pairs_0_dict[3]=torch.empty((0, 2), dtype=torch.int32)
        self.pairs_0_dict[4]=torch.tensor([[2, 4]], dtype=torch.int32)
        self.pairs_0_dict[5]=torch.tensor([[3, 5]], dtype=torch.int32)
        self.pairs_0_dict[6]=torch.tensor([[2, 5]], dtype=torch.int32)
        self.pairs_0_dict[7]=torch.tensor([[2, 4]], dtype=torch.int32)
        self.pairs_0_dict[8]=torch.tensor([[2, 5]], dtype=torch.int32)
        self.pairs_0_dict[9]=torch.tensor([[3, 5]], dtype=torch.int32)
        self.pairs_0_dict[10]=torch.tensor([[3, 6]], dtype=torch.int32)
        self.pairs_0_dict[11]=torch.tensor([[3, 6]], dtype=torch.int32)

        self.lengths_trees=[4,5,5,6,6,7,7,7,8,8,8,9]

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, inputs):
        x, tree_topologies = inputs  # Unpack tuple
        # Apply convolution separately on patch size 2
        outs = []
        for t_type in range(12):
            kk_0 = self.kk_0_dict[t_type]
            # print(kk_0)
            kk_0 = torch.tensor(kk_0, dtype=torch.int)  # Shape: (num_kk,)
            # print(out_d.shape)
            dp_1 = kk_0  # Shape: (num_kk,)
            length = self.lengths_trees[t_type]
            dp_2 = length - kk_0  # Shape: (num_kk,)
            full_filter = torch.zeros(
                self.filter_2.shape[0],  # Out channels
                self.filter_2.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_2.device,
            )
            #out_d = torch.zeros(
             #   x.shape[0], self.filter_2.shape[0], self.n - length + 1, device=x.device
            #)

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0
            indices = torch.stack(
                [dp_1 - self.min_split, self.n_span * dp_1 + dp_2 - self.min_split], dim=1
            ).permute(1,0)  # Shape: (2,num_kk)

            num_filters_2 = indices.shape[1]

            # Replicate full_filter across a new dimension for the filters.
            stacked_filters_2 = full_filter.unsqueeze(-1).expand(-1, -1, -1, num_filters_2).clone()
            # Create filter indices for the last dimension.
            filter_indices = torch.arange(num_filters_2)

            # Expand filter_2 slices to shape [7,6,num_filters]
            filter_2_slice0 = self.filter_2[:, :, 0].unsqueeze(-1).expand(self.filter_2.size(0), self.filter_2.size(1), num_filters_2)
            filter_2_slice1 = self.filter_2[:, :, 1].unsqueeze(-1).expand(self.filter_2.size(0), self.filter_2.size(1), num_filters_2)

            # Update the positions specified by indices for each filter.
            stacked_filters_2[:, :, indices[0, :], filter_indices] = filter_2_slice0/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )
            stacked_filters_2[:, :, indices[1, :], filter_indices] = filter_2_slice1/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )

            # Reshape the stacked filters to shape [num_filters_2, 7, 6, 10]

            stacked_filters_2=stacked_filters_2.permute(3,0,1,2)


            #out_d+=manual_conv1d_stacked(x, stacked_filters, stride=self.stride)/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )
            
            #Ternary filters

            pairs_0 = self.pairs_0_dict[t_type]
            #pairs_0 = torch.tensor(pairs_0, dtype=torch.int)  # Shape: (num_kk,2)
            full_filter = torch.zeros(
                self.filter_3.shape[0],  # Out channels
                self.filter_3.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_3.device,
            )

            dp_1 = pairs_0[:, 0]
            dp_2 = pairs_0[:, 1] - pairs_0[:, 0]
            dp_3 = length - pairs_0[:, 1]

            indices = torch.stack(
                [
                    dp_1 - self.min_split,
                    self.n_span * dp_1 + dp_2 - self.min_split,
                    self.n_span * (dp_1 + dp_2) + dp_3 - self.min_split,
                ],
                dim=1,
            ).permute(
                1, 0
            )  # Shape: (3,num_kk)

            num_filters_3 = indices.shape[1]

            # Replicate full_filter across a new dimension for the filters.
            stacked_filters_3 = full_filter.unsqueeze(-1).expand(-1, -1, -1, num_filters_3).clone()
            # Create filter indices for the last dimension.
            filter_indices = torch.arange(num_filters_3)

            # Expand filter_2 slices to shape [7,6,num_filters]
            filter_3_slice0 = self.filter_3[:, :, 0].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)
            filter_3_slice1 = self.filter_3[:, :, 1].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)
            filter_3_slice2 = self.filter_3[:, :, 2].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)

            # Update the positions specified by indices for each filter.
            stacked_filters_3[:, :, indices[0, :], filter_indices] = filter_3_slice0/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )
            stacked_filters_3[:, :, indices[1, :], filter_indices] = filter_3_slice1/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )
            stacked_filters_3[:, :, indices[2, :], filter_indices] = filter_3_slice2/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )

            # Reshape the stacked filters to shape [num_filters, 7, 6, 10]

            stacked_filters_3=stacked_filters_3.permute(3,0,1,2) # Shape: (num_filters_3, out_channels, in_channels, kernel_size)
            
            stacked_filters = torch.cat((stacked_filters_2, stacked_filters_3), dim=0)
            
            out_d=manual_conv1d_stacked(x, stacked_filters, stride=self.stride)

            #pad = torch.zeros(
             #   out_d.shape[0],
              #  out_d.shape[1],
               # self.n - out_d.shape[2],
                #device=out_d.device,
                #dtype=out_d.dtype,
            #)
            #out_d_padded = torch.cat([out_d, pad], dim=2)
            outs.append(out_d)
        # Step 1: Stack all elements in outs along a new dimension (dim=3)
        intermediate = torch.stack(
            outs, dim=3
        )  # Shape: (batch_size, out_channels, seq_len, num_spans)

        # Step 2: Reshape to interleave
        interleaved_out = intermediate.view(
            intermediate.shape[0], intermediate.shape[1], -1
        )

        return interleaved_out, tree_topologies  # Return tuple







class MyConv1d_ell_4_last_parallel(nn.Module):
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

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        l = 4
        self.n = 81
        self.min_d = 2**l
        self.max_d = 3 ** (l)
        self.min_split = 2 ** (l - 1)
        self.max_split = 3 ** (l - 1)
        self.n_span = 3 ** (l - 1) - 2 ** (l - 1) + 1
        # Dictionary to store kk_0 lists
        self.stride = self.n_span*self.n

        self.kk_0_dict = {}
        self.pairs_0_dict = {}

        for length in range(self.min_d, self.max_d + 1):  # Length of the span
            kk_0 = np.array(range(self.min_split, length - self.min_split + 1))
            kk_0 = kk_0[(kk_0 <= self.max_split) & (kk_0 >= (length - self.max_split))]
            self.kk_0_dict[length] = kk_0.tolist()
            pairs_0 = nested_ranges_as_tensor(0, length, self.min_split)
            condition_1 = pairs_0[:, 0] <= self.max_split
            condition_2 = ((pairs_0[:, 1] - pairs_0[:, 0]) <= self.max_split) & (
                (pairs_0[:, 1] - pairs_0[:, 0]) >= self.min_split
            )
            condition_3 = (pairs_0[:, 1] >= (length - self.max_split)) & (
                length - pairs_0[:, 1] >= self.min_split
            )
            # Combine all conditions
            pairs_0 = pairs_0[condition_1 & condition_2 & condition_3]
            self.pairs_0_dict[length] = pairs_0

        # Pruning condition: keep elements <= max_split and >= (length - max_split)

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, inputs):
        x, sum_of_squares = inputs  # Unpack tuple
        outs = []

        # 1. Get unique lengths actually present
        #present_lengths = torch.unique(sum_of_squares.flatten())
        for length in range(self.min_d, self.max_d + 1):
            #length = int(length.item())
            kk_0 = self.kk_0_dict[length]
            # print(kk_0)
            out_d = torch.zeros(
                x.shape[0], self.filter_2.shape[0], 1, device=x.device
            )
            # print(out_d.shape)
            kk_0 = torch.tensor(kk_0, dtype=torch.int)  # Shape: (num_kk,)
            dp_1 = kk_0
            dp_2 = length - kk_0

            pos1 = dp_1 - self.min_split
            pos2 = self.n_span * dp_1 + dp_2 - self.min_split

            patch1 = x[:, :, pos1]  # [batch_size, in_channels, num_kk]
            patch2 = x[:, :, pos2]  # [batch_size, in_channels, num_kk]

            patch = torch.stack([patch1, patch2], dim=3)  # [batch_size, in_channels, num_kk, 2]

            intermediate = torch.einsum('bick,oik->boc', patch, self.filter_2) / (self.filter_2.size(1) * self.filter_2.size(2))**0.5

            out_d += intermediate.sum(dim=2, keepdim=True)  # [batch_size, out_channels, 1]

                # Possibly normalize
            pairs_0 = self.pairs_0_dict[length]
            dp_1 = pairs_0[:, 0]
            dp_2 = pairs_0[:, 1] - pairs_0[:, 0]
            dp_3 = length - pairs_0[:, 1]


            patch1 = x[:, :, dp_1 - self.min_split]
            patch2 = x[:, :, self.n_span * dp_1 + dp_2 - self.min_split]
            patch3 = x[:, :, self.n_span * (dp_1 + dp_2) + dp_3 - self.min_split]

            # Stack or concatenate them
            patch = torch.stack([patch1, patch2, patch3], dim=3)  # Shape: [batch_size, in_channels,num_kk, 3]

            # Now apply the 1x2 convolution manually:
            # Shape of filter_2: [out_channels, in_channels, 2]
            intermediate = torch.einsum('bick,oik->boc', patch, self.filter_3) / (self.filter_3.size(1) * self.filter_3.size(2))**0.5

            out_d += intermediate.sum(dim=2, keepdim=True)  # [batch_size, out_channels, 1]
            
        
            outs.append(out_d)
        #print(outs.shape)
        # Step 1: Stack all elements in outs along a new dimension (dim=3)
        intermediate = torch.stack(
            outs, dim=3
        )  # Shape: (batch_size, out_channels, seq_len, num_filters)

        # Step 2: Reshape to interleave
        interleaved_out = intermediate.view(
            intermediate.shape[0], intermediate.shape[1], -1
        )

        return interleaved_out, sum_of_squares  # Return tuple

















class MyConv1d_ell_4_last(nn.Module):
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

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        l = 4
        self.n = 81
        self.min_d = 2**l
        self.max_d = 3 ** (l)
        self.min_split = 2 ** (l - 1)
        self.max_split = 3 ** (l - 1)
        self.n_span = 3 ** (l - 1) - 2 ** (l - 1) + 1
        # Dictionary to store kk_0 lists
        self.stride = self.n_span*self.n

        self.kk_0_dict = {}
        self.pairs_0_dict = {}

        for length in range(self.min_d, self.max_d + 1):  # Length of the span
            kk_0 = np.array(range(self.min_split, length - self.min_split + 1))
            kk_0 = kk_0[(kk_0 <= self.max_split) & (kk_0 >= (length - self.max_split))]
            self.kk_0_dict[length] = kk_0.tolist()
            pairs_0 = nested_ranges_as_tensor(0, length, self.min_split)
            condition_1 = pairs_0[:, 0] <= self.max_split
            condition_2 = ((pairs_0[:, 1] - pairs_0[:, 0]) <= self.max_split) & (
                (pairs_0[:, 1] - pairs_0[:, 0]) >= self.min_split
            )
            condition_3 = (pairs_0[:, 1] >= (length - self.max_split)) & (
                length - pairs_0[:, 1] >= self.min_split
            )
            # Combine all conditions
            pairs_0 = pairs_0[condition_1 & condition_2 & condition_3]
            self.pairs_0_dict[length] = pairs_0

        # Pruning condition: keep elements <= max_split and >= (length - max_split)

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, inputs):
        x, sum_of_squares = inputs  # Unpack tuple
        # Apply convolution separately on patch size 2
        outs = []
        



        for length in range(self.min_d, self.max_d + 1):
            kk_0 = self.kk_0_dict[length]
            # print(kk_0)
            kk_0 = torch.tensor(kk_0, dtype=torch.int)  # Shape: (num_kk,)
            # print(out_d.shape)
            dp_1 = kk_0  # Shape: (num_kk,)
            dp_2 = length - kk_0  # Shape: (num_kk,)
            full_filter = torch.zeros(
                self.filter_2.shape[0],  # Out channels
                self.filter_2.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_2.device,
            )
            out_d = torch.zeros(
                x.shape[0], self.filter_2.shape[0], 1, device=x.device
            )

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0
            indices = torch.stack(
                [dp_1 - self.min_split, self.n_span * dp_1 + dp_2 - self.min_split], dim=1
            ).permute(1,0)  # Shape: (2,num_kk)


            num_filters_2 = indices.shape[1]


            group_size = 2  # << You can adjust this easily!

            for start_idx in range(0, num_filters_2, group_size):
                end_idx = min(start_idx + group_size, num_filters_2)
                
                # Slice the indices for this group
                indices_group = indices[:, start_idx:end_idx]
                current_group_size = end_idx - start_idx  # may be smaller at the last group

                # Create the stacked_filters for this group
                stacked_filters = full_filter.unsqueeze(-1).expand(-1, -1, -1, current_group_size).clone()
                filter_indices = torch.arange(current_group_size, device=indices.device)

                filter_2_slice0 = self.filter_2[:, :, 0].unsqueeze(-1).expand(self.filter_2.shape[0], self.filter_2.shape[1], current_group_size)
                filter_2_slice1 = self.filter_2[:, :, 1].unsqueeze(-1).expand(self.filter_2.shape[0], self.filter_2.shape[1], current_group_size)

                stacked_filters[:, :, indices_group[0, :], filter_indices] = filter_2_slice0 / ((full_filter.size(1) * self.filter_2.size(2)) ** 0.5)
                stacked_filters[:, :, indices_group[1, :], filter_indices] = filter_2_slice1 / ((full_filter.size(1) * self.filter_2.size(2)) ** 0.5)

                stacked_filters = stacked_filters.permute(3, 0, 1, 2)  # (current_group_size, out_channels, in_channels, kernel_size)

                # Apply convolution and accumulate the result
                out_d += manual_conv1d_stacked(x, stacked_filters, stride=self.stride)



            
            #Ternary filters

            pairs_0 = self.pairs_0_dict[length]
            #pairs_0 = torch.tensor(pairs_0, dtype=torch.int)  # Shape: (num_kk,2)
            full_filter = torch.zeros(
                self.filter_3.shape[0],  # Out channels
                self.filter_3.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_3.device,
            )

            dp_1 = pairs_0[:, 0]
            dp_2 = pairs_0[:, 1] - pairs_0[:, 0]
            dp_3 = length - pairs_0[:, 1]

            indices = torch.stack(
                [
                    dp_1 - self.min_split,
                    self.n_span * dp_1 + dp_2 - self.min_split,
                    self.n_span * (dp_1 + dp_2) + dp_3 - self.min_split,
                ],
                dim=1,
            ).permute(
                1, 0
            )  # Shape: (3,num_kk)

            num_filters_3 = indices.shape[1]

            group_size = 2  # << You can adjust this easily!

            for start_idx in range(0, num_filters_3, group_size):
                end_idx = min(start_idx + group_size, num_filters_3)
                
                # Slice the indices for this group
                indices_group = indices[:, start_idx:end_idx]
                current_group_size = end_idx - start_idx  # may be smaller at the last group

                # Create the stacked_filters for this group
                stacked_filters = full_filter.unsqueeze(-1).expand(-1, -1, -1, current_group_size).clone()
                filter_indices = torch.arange(current_group_size, device=indices.device)

                filter_3_slice0 = self.filter_3[:, :, 0].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], current_group_size)
                filter_3_slice1 = self.filter_3[:, :, 1].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], current_group_size)
                filter_3_slice2 = self.filter_3[:, :, 2].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], current_group_size)

                stacked_filters[:, :, indices_group[0, :], filter_indices] = filter_3_slice0 / ((full_filter.size(1) * self.filter_3.size(2)) ** 0.5)
                stacked_filters[:, :, indices_group[1, :], filter_indices] = filter_3_slice1 / ((full_filter.size(1) * self.filter_3.size(2)) ** 0.5)
                stacked_filters[:, :, indices_group[2, :], filter_indices] = filter_3_slice2 / ((full_filter.size(1) * self.filter_3.size(2)) ** 0.5)

                stacked_filters = stacked_filters.permute(3, 0, 1, 2)  # (current_group_size, out_channels, in_channels, kernel_size)

                # Apply convolution and accumulate the result
                out_d += manual_conv1d_stacked(x, stacked_filters, stride=self.stride)






            
            #pad = torch.zeros(
             #   out_d.shape[0],
              #  out_d.shape[1],
               # self.n - out_d.shape[2],
                #device=out_d.device,
                #dtype=out_d.dtype,
            #)
            #out_d_padded = torch.cat([out_d, pad], dim=2)
            outs.append(out_d)
        # Step 1: Stack all elements in outs along a new dimension (dim=3)
        intermediate = torch.stack(
            outs, dim=3
        )  # Shape: (batch_size, out_channels, seq_len, num_spans)

        # Step 2: Reshape to interleave
        interleaved_out = intermediate.view(
            intermediate.shape[0], intermediate.shape[1], -1
        )

        return interleaved_out, sum_of_squares  # Return tuple

class MyConv1d_ell_4_last_old(nn.Module):
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

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        l = 4
        self.n = 81
        self.min_d = 2**l
        self.max_d = 3 ** (l)
        self.min_split = 2 ** (l - 1)
        self.max_split = 3 ** (l - 1)
        self.n_span = 3 ** (l - 1) - 2 ** (l - 1) + 1
        # Dictionary to store kk_0 lists
        self.stride = self.n_span*self.n

        self.kk_0_dict = {}
        self.pairs_0_dict = {}

        for length in range(self.min_d, self.max_d + 1):  # Length of the span
            kk_0 = np.array(range(self.min_split, length - self.min_split + 1))
            kk_0 = kk_0[(kk_0 <= self.max_split) & (kk_0 >= (length - self.max_split))]
            self.kk_0_dict[length] = kk_0.tolist()
            pairs_0 = nested_ranges_as_tensor(0, length, self.min_split)
            condition_1 = pairs_0[:, 0] <= self.max_split
            condition_2 = ((pairs_0[:, 1] - pairs_0[:, 0]) <= self.max_split) & (
                (pairs_0[:, 1] - pairs_0[:, 0]) >= self.min_split
            )
            condition_3 = (pairs_0[:, 1] >= (length - self.max_split)) & (
                length - pairs_0[:, 1] >= self.min_split
            )
            # Combine all conditions
            pairs_0 = pairs_0[condition_1 & condition_2 & condition_3]
            self.pairs_0_dict[length] = pairs_0

        # Pruning condition: keep elements <= max_split and >= (length - max_split)

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, inputs):
        x, sum_of_squares = inputs  # Unpack tuple 
        # Apply convolution separately on patch size 2
        outs = []
        for length in range(self.min_d, self.max_d + 1):
            kk_0 = self.kk_0_dict[length]
            # print(kk_0)
            kk_0 = torch.tensor(kk_0, dtype=torch.int)  # Shape: (num_kk,)
            # print(out_d.shape)
            dp_1 = kk_0  # Shape: (num_kk,)
            dp_2 = length - kk_0  # Shape: (num_kk,)
            full_filter = torch.zeros(
                self.filter_2.shape[0],  # Out channels
                self.filter_2.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_2.device,
            )
            #out_d = torch.zeros(
             #   x.shape[0], self.filter_2.shape[0], self.n - length + 1, device=x.device
            #)

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0
            indices = torch.stack(
                [dp_1 - self.min_split, self.n_span * dp_1 + dp_2 - self.min_split], dim=1
            ).permute(1,0)  # Shape: (2,num_kk)

            num_filters_2 = indices.shape[1]

            # Replicate full_filter across a new dimension for the filters.
            stacked_filters_2 = full_filter.unsqueeze(-1).expand(-1, -1, -1, num_filters_2).clone()
            # Create filter indices for the last dimension.
            filter_indices = torch.arange(num_filters_2)

            # Expand filter_2 slices to shape [7,6,num_filters]
            filter_2_slice0 = self.filter_2[:, :, 0].unsqueeze(-1).expand(self.filter_2.size(0), self.filter_2.size(1), num_filters_2)
            filter_2_slice1 = self.filter_2[:, :, 1].unsqueeze(-1).expand(self.filter_2.size(0), self.filter_2.size(1), num_filters_2)

            # Update the positions specified by indices for each filter.
            stacked_filters_2[:, :, indices[0, :], filter_indices] = filter_2_slice0/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )
            stacked_filters_2[:, :, indices[1, :], filter_indices] = filter_2_slice1/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )

            # Reshape the stacked filters to shape [num_filters_2, 7, 6, 10]

            stacked_filters_2=stacked_filters_2.permute(3,0,1,2)


            #out_d+=manual_conv1d_stacked(x, stacked_filters, stride=self.stride)/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )
            
            #Ternary filters

            pairs_0 = self.pairs_0_dict[length]
            #pairs_0 = torch.tensor(pairs_0, dtype=torch.int)  # Shape: (num_kk,2)
            full_filter = torch.zeros(
                self.filter_3.shape[0],  # Out channels
                self.filter_3.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_3.device,
            )

            dp_1 = pairs_0[:, 0]
            dp_2 = pairs_0[:, 1] - pairs_0[:, 0]
            dp_3 = length - pairs_0[:, 1]

            indices = torch.stack(
                [
                    dp_1 - self.min_split,
                    self.n_span * dp_1 + dp_2 - self.min_split,
                    self.n_span * (dp_1 + dp_2) + dp_3 - self.min_split,
                ],
                dim=1,
            ).permute(
                1, 0
            )  # Shape: (3,num_kk)

            num_filters_3 = indices.shape[1]

            # Replicate full_filter across a new dimension for the filters.
            stacked_filters_3 = full_filter.unsqueeze(-1).expand(-1, -1, -1, num_filters_3).clone()
            # Create filter indices for the last dimension.
            filter_indices = torch.arange(num_filters_3)

            # Expand filter_2 slices to shape [7,6,num_filters]
            filter_3_slice0 = self.filter_3[:, :, 0].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)
            filter_3_slice1 = self.filter_3[:, :, 1].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)
            filter_3_slice2 = self.filter_3[:, :, 2].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)

            # Update the positions specified by indices for each filter.
            stacked_filters_3[:, :, indices[0, :], filter_indices] = filter_3_slice0/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )
            stacked_filters_3[:, :, indices[1, :], filter_indices] = filter_3_slice1/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )
            stacked_filters_3[:, :, indices[2, :], filter_indices] = filter_3_slice2/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )

            # Reshape the stacked filters to shape [num_filters, 7, 6, 10]

            stacked_filters_3=stacked_filters_3.permute(3,0,1,2) # Shape: (num_filters_3, out_channels, in_channels, kernel_size)
            
            stacked_filters = torch.cat((stacked_filters_2, stacked_filters_3), dim=0)
            
            out_d=manual_conv1d_stacked(x, stacked_filters, stride=self.stride)

            #pad = torch.zeros(
             #   out_d.shape[0],
              #  out_d.shape[1],
               # self.n - out_d.shape[2],
                #device=out_d.device,
                #dtype=out_d.dtype,
            #)
            #out_d_padded = torch.cat([out_d, pad], dim=2)
            outs.append(out_d)
        # Step 1: Stack all elements in outs along a new dimension (dim=3)
        intermediate = torch.stack(
            outs, dim=3
        )  # Shape: (batch_size, out_channels, seq_len, num_spans)

        # Step 2: Reshape to interleave
        interleaved_out = intermediate.view(
            intermediate.shape[0], intermediate.shape[1], -1
        )

        return interleaved_out, sum_of_squares  # Return tuple



class MyConv1d_ell_2(nn.Module):
    def __init__(self, in_channels, out_channels,len_seq, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include a bias term
        """
        super().__init__()
        self.filter_size_2 = 2
        self.filter_size_3 = 3

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        l = 2
        self.n = len_seq
        self.min_d = 2**l
        self.max_d = 3 ** (l)
        self.min_split = 2 ** (l - 1)
        self.max_split = 3 ** (l - 1)
        self.n_span = 3 ** (l - 1) - 2 ** (l - 1) + 1
        # Dictionary to store kk_0 lists
        self.stride = self.n_span

        self.kk_0_dict = {}
        self.pairs_0_dict = {}

        for length in range(self.min_d, self.max_d + 1):  # Length of the span
            kk_0 = np.array(range(self.min_split, length - self.min_split + 1))
            kk_0 = kk_0[(kk_0 <= self.max_split) & (kk_0 >= (length - self.max_split))]
            self.kk_0_dict[length] = kk_0.tolist()
            pairs_0 = nested_ranges_as_tensor(0, length, self.min_split)
            condition_1 = pairs_0[:, 0] <= self.max_split
            condition_2 = ((pairs_0[:, 1] - pairs_0[:, 0]) <= self.max_split) & (
                (pairs_0[:, 1] - pairs_0[:, 0]) >= self.min_split
            )
            condition_3 = (pairs_0[:, 1] >= (length - self.max_split)) & (
                length - pairs_0[:, 1] >= self.min_split
            )
            # Combine all conditions
            pairs_0 = pairs_0[condition_1 & condition_2 & condition_3]
            self.pairs_0_dict[length] = pairs_0

        # Pruning condition: keep elements <= max_split and >= (length - max_split)

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, inputs):
        x, sum_of_squares = inputs  # Unpack tuple
        # Apply convolution separately on patch size 2
        outs = []
        for length in range(self.min_d, self.max_d + 1):
            kk_0 = self.kk_0_dict[length]
            # print(kk_0)
            kk_0 = torch.tensor(kk_0, dtype=torch.int)  # Shape: (num_kk,)
            # print(out_d.shape)
            dp_1 = kk_0  # Shape: (num_kk,)
            dp_2 = length - kk_0  # Shape: (num_kk,)
            full_filter = torch.zeros(
                self.filter_2.shape[0],  # Out channels
                self.filter_2.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_2.device,
            )
            #out_d = torch.zeros(
             #   x.shape[0], self.filter_2.shape[0], self.n - length + 1, device=x.device
            #)

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0
            indices = torch.stack(
                [dp_1 - self.min_split, self.n_span * dp_1 + dp_2 - self.min_split], dim=1
            ).permute(1,0)  # Shape: (2,num_kk)

            num_filters_2 = indices.shape[1]

            # Replicate full_filter across a new dimension for the filters.
            stacked_filters_2 = full_filter.unsqueeze(-1).expand(-1, -1, -1, num_filters_2).clone()
            # Create filter indices for the last dimension.
            filter_indices = torch.arange(num_filters_2)

            # Expand filter_2 slices to shape [7,6,num_filters]
            filter_2_slice0 = self.filter_2[:, :, 0].unsqueeze(-1).expand(self.filter_2.size(0), self.filter_2.size(1), num_filters_2)
            filter_2_slice1 = self.filter_2[:, :, 1].unsqueeze(-1).expand(self.filter_2.size(0), self.filter_2.size(1), num_filters_2)

            # Update the positions specified by indices for each filter.
            stacked_filters_2[:, :, indices[0, :], filter_indices] = filter_2_slice0/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )
            stacked_filters_2[:, :, indices[1, :], filter_indices] = filter_2_slice1/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )

            # Reshape the stacked filters to shape [num_filters_2, 7, 6, 10]

            stacked_filters_2=stacked_filters_2.permute(3,0,1,2)


            #out_d+=manual_conv1d_stacked(x, stacked_filters, stride=self.stride)/((full_filter.size(1) * self.filter_2.size(2)) ** 0.5 )
            
            #Ternary filters

            pairs_0 = self.pairs_0_dict[length]
            #pairs_0 = torch.tensor(pairs_0, dtype=torch.int)  # Shape: (num_kk,2)
            full_filter = torch.zeros(
                self.filter_3.shape[0],  # Out channels
                self.filter_3.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_3.device,
            )

            dp_1 = pairs_0[:, 0]
            dp_2 = pairs_0[:, 1] - pairs_0[:, 0]
            dp_3 = length - pairs_0[:, 1]

            indices = torch.stack(
                [
                    dp_1 - self.min_split,
                    self.n_span * dp_1 + dp_2 - self.min_split,
                    self.n_span * (dp_1 + dp_2) + dp_3 - self.min_split,
                ],
                dim=1,
            ).permute(
                1, 0
            )  # Shape: (3,num_kk)

            num_filters_3 = indices.shape[1]

            # Replicate full_filter across a new dimension for the filters.
            stacked_filters_3 = full_filter.unsqueeze(-1).expand(-1, -1, -1, num_filters_3).clone()
            # Create filter indices for the last dimension.
            filter_indices = torch.arange(num_filters_3)

            # Expand filter_2 slices to shape [7,6,num_filters]
            filter_3_slice0 = self.filter_3[:, :, 0].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)
            filter_3_slice1 = self.filter_3[:, :, 1].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)
            filter_3_slice2 = self.filter_3[:, :, 2].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], num_filters_3)

            # Update the positions specified by indices for each filter.
            stacked_filters_3[:, :, indices[0, :], filter_indices] = filter_3_slice0/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )
            stacked_filters_3[:, :, indices[1, :], filter_indices] = filter_3_slice1/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )
            stacked_filters_3[:, :, indices[2, :], filter_indices] = filter_3_slice2/((full_filter.size(1) * self.filter_3.size(2)) ** 0.5 )

            # Reshape the stacked filters to shape [num_filters, 7, 6, 10]

            stacked_filters_3=stacked_filters_3.permute(3,0,1,2) # Shape: (num_filters_3, out_channels, in_channels, kernel_size)
            
            stacked_filters = torch.cat((stacked_filters_2, stacked_filters_3), dim=0)
            
            out_d=manual_conv1d_stacked(x, stacked_filters, stride=self.stride)

            pad = torch.zeros(
                out_d.shape[0],
                out_d.shape[1],
                self.n - out_d.shape[2],
                device=out_d.device,
                dtype=out_d.dtype,
            )
            out_d_padded = torch.cat([out_d, pad], dim=2)
            outs.append(out_d_padded)
        # Step 1: Stack all elements in outs along a new dimension (dim=3)
        intermediate = torch.stack(
            outs, dim=3
        )  # Shape: (batch_size, out_channels, seq_len, num_filters)

        # Step 2: Reshape to interleave
        interleaved_out = intermediate.view(
            intermediate.shape[0], intermediate.shape[1], -1
        )

        return interleaved_out, sum_of_squares  # Return tuple
    


class MyConv1d_ell_3(nn.Module):
    def __init__(self, in_channels, out_channels,len_seq, bias=False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include a bias term
        """
        super().__init__()
        self.filter_size_2 = 2
        self.filter_size_3 = 3

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        l = 3
        self.n = len_seq
        self.min_d = 2**l
        self.max_d = 3 ** (l)
        self.min_split = 2 ** (l - 1)
        self.max_split = 3 ** (l - 1)
        self.n_span = 3 ** (l - 1) - 2 ** (l - 1) + 1
        # Dictionary to store kk_0 lists
        self.stride = self.n_span

        self.kk_0_dict = {}
        self.pairs_0_dict = {}

        for length in range(self.min_d, self.max_d + 1):  # Length of the span
            kk_0 = np.array(range(self.min_split, length - self.min_split + 1))
            kk_0 = kk_0[(kk_0 <= self.max_split) & (kk_0 >= (length - self.max_split))]
            self.kk_0_dict[length] = kk_0.tolist()
            pairs_0 = nested_ranges_as_tensor(0, length, self.min_split)
            condition_1 = pairs_0[:, 0] <= self.max_split
            condition_2 = ((pairs_0[:, 1] - pairs_0[:, 0]) <= self.max_split) & (
                (pairs_0[:, 1] - pairs_0[:, 0]) >= self.min_split
            )
            condition_3 = (pairs_0[:, 1] >= (length - self.max_split)) & (
                length - pairs_0[:, 1] >= self.min_split
            )
            # Combine all conditions
            pairs_0 = pairs_0[condition_1 & condition_2 & condition_3]
            self.pairs_0_dict[length] = pairs_0

        # Pruning condition: keep elements <= max_split and >= (length - max_split)

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, inputs):
        x, sum_of_squares = inputs  # Unpack tuple
        # Apply convolution separately on patch size 2
        outs = []
        for length in range(self.min_d, self.max_d + 1):
            kk_0 = self.kk_0_dict[length]
            # print(kk_0)
            kk_0 = torch.tensor(kk_0, dtype=torch.int)  # Shape: (num_kk,)
            # print(out_d.shape)
            dp_1 = kk_0  # Shape: (num_kk,)
            dp_2 = length - kk_0  # Shape: (num_kk,)
            full_filter = torch.zeros(
                self.filter_2.shape[0],  # Out channels
                self.filter_2.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_2.device,
            )
            out_d = torch.zeros(
                x.shape[0], self.filter_2.shape[0], self.n - length + 1, device=x.device
            )

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0

            # Compute indices for efficient summation
            dp_1 = kk_0
            dp_2 = length - kk_0
            indices = torch.stack(
                [dp_1 - self.min_split, self.n_span * dp_1 + dp_2 - self.min_split], dim=1
            ).permute(1,0)  # Shape: (2,num_kk)

            num_filters_2 = indices.shape[1]


            group_size = 1  # << You can adjust this easily!

            for start_idx in range(0, num_filters_2, group_size):
                end_idx = min(start_idx + group_size, num_filters_2)
                
                # Slice the indices for this group
                indices_group = indices[:, start_idx:end_idx]
                current_group_size = end_idx - start_idx  # may be smaller at the last group

                # Create the stacked_filters for this group
                stacked_filters = full_filter.unsqueeze(-1).expand(-1, -1, -1, current_group_size).clone()
                filter_indices = torch.arange(current_group_size, device=indices.device)

                filter_2_slice0 = self.filter_2[:, :, 0].unsqueeze(-1).expand(self.filter_2.shape[0], self.filter_2.shape[1], current_group_size)
                filter_2_slice1 = self.filter_2[:, :, 1].unsqueeze(-1).expand(self.filter_2.shape[0], self.filter_2.shape[1], current_group_size)

                stacked_filters[:, :, indices_group[0, :], filter_indices] = filter_2_slice0 / ((full_filter.size(1) * self.filter_2.size(2)) ** 0.5)
                stacked_filters[:, :, indices_group[1, :], filter_indices] = filter_2_slice1 / ((full_filter.size(1) * self.filter_2.size(2)) ** 0.5)

                stacked_filters = stacked_filters.permute(3, 0, 1, 2)  # (current_group_size, out_channels, in_channels, kernel_size)

                # Apply convolution and accumulate the result
                out_d += manual_conv1d_stacked(x, stacked_filters, stride=self.stride)

            
            
            #Ternary filters

            pairs_0 = self.pairs_0_dict[length]
            #pairs_0 = torch.tensor(pairs_0, dtype=torch.int)  # Shape: (num_kk,2)
            full_filter = torch.zeros(
                self.filter_3.shape[0],  # Out channels
                self.filter_3.shape[1],  # In channels
                length * self.n_span,  # Kernel size
                device=self.filter_3.device,
            )

            dp_1 = pairs_0[:, 0]
            dp_2 = pairs_0[:, 1] - pairs_0[:, 0]
            dp_3 = length - pairs_0[:, 1]

            indices = torch.stack(
                [
                    dp_1 - self.min_split,
                    self.n_span * dp_1 + dp_2 - self.min_split,
                    self.n_span * (dp_1 + dp_2) + dp_3 - self.min_split,
                ],
                dim=1,
            ).permute(
                1, 0
            )  # Shape: (3,num_kk)

            num_filters_3 = indices.shape[1]


            

            group_size = 1  # << You can adjust this easily!

            for start_idx in range(0, num_filters_3, group_size):
                end_idx = min(start_idx + group_size, num_filters_3)
                
                # Slice the indices for this group
                indices_group = indices[:, start_idx:end_idx]
                current_group_size = end_idx - start_idx  # may be smaller at the last group

                # Create the stacked_filters for this group
                stacked_filters = full_filter.unsqueeze(-1).expand(-1, -1, -1, current_group_size).clone()
                filter_indices = torch.arange(current_group_size, device=indices.device)

                filter_3_slice0 = self.filter_3[:, :, 0].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], current_group_size)
                filter_3_slice1 = self.filter_3[:, :, 1].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], current_group_size)
                filter_3_slice2 = self.filter_3[:, :, 2].unsqueeze(-1).expand(self.filter_3.shape[0], self.filter_3.shape[1], current_group_size)

                stacked_filters[:, :, indices_group[0, :], filter_indices] = filter_3_slice0 / ((full_filter.size(1) * self.filter_3.size(2)) ** 0.5)
                stacked_filters[:, :, indices_group[1, :], filter_indices] = filter_3_slice1 / ((full_filter.size(1) * self.filter_3.size(2)) ** 0.5)
                stacked_filters[:, :, indices_group[2, :], filter_indices] = filter_3_slice2 / ((full_filter.size(1) * self.filter_3.size(2)) ** 0.5)

                stacked_filters = stacked_filters.permute(3, 0, 1, 2)  # (current_group_size, out_channels, in_channels, kernel_size)

                # Apply convolution and accumulate the result
                out_d += manual_conv1d_stacked(x, stacked_filters, stride=self.stride)



            pad = torch.zeros(
                out_d.shape[0],
                out_d.shape[1],
                self.n - out_d.shape[2],
                device=out_d.device,
                dtype=out_d.dtype,
            )
            out_d_padded = torch.cat([out_d, pad], dim=2)
            outs.append(out_d_padded)
        # Step 1: Stack all elements in outs along a new dimension (dim=3)
        intermediate = torch.stack(
            outs, dim=3
        )  # Shape: (batch_size, out_channels, seq_len, num_filters)

        # Step 2: Reshape to interleave
        interleaved_out = intermediate.view(
            intermediate.shape[0], intermediate.shape[1], -1
        )

        return interleaved_out, sum_of_squares  # Return tuple


class MyConv1d_ell_2_sequential(nn.Module):
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

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        l = 2
        self.n = 9
        self.min_d = 2**l
        self.max_d = 3 ** (l)
        self.min_split = 2 ** (l - 1)
        self.max_split = 3 ** (l - 1)
        self.n_span = 3 ** (l - 1) - 2 ** (l - 1) + 1
        # Dictionary to store kk_0 lists
        #self.stride = self.n_span
        self.stride=1000

        self.kk_0_dict = {}
        self.pairs_0_dict = {}

        for length in range(self.min_d, self.max_d + 1):  # Length of the span
            kk_0 = np.array(range(self.min_split, length - self.min_split + 1))
            kk_0 = kk_0[(kk_0 <= self.max_split) & (kk_0 >= (length - self.max_split))]
            self.kk_0_dict[length] = kk_0.tolist()
            pairs_0 = nested_ranges_as_tensor(0, length, self.min_split)
            condition_1 = pairs_0[:, 0] <= self.max_split
            condition_2 = ((pairs_0[:, 1] - pairs_0[:, 0]) <= self.max_split) & (
                (pairs_0[:, 1] - pairs_0[:, 0]) >= self.min_split
            )
            condition_3 = (pairs_0[:, 1] >= (length - self.max_split)) & (
                length - pairs_0[:, 1] >= self.min_split
            )
            # Combine all conditions
            pairs_0 = pairs_0[condition_1 & condition_2 & condition_3]
            self.pairs_0_dict[length] = pairs_0

        # Pruning condition: keep elements <= max_split and >= (length - max_split)

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, inputs):
        x, sum_of_squares = inputs  # Unpack tuple
        # Apply convolution separately on patch size 2
        outs = []
        for length in range(self.min_d, self.max_d + 1):
            kk_0 = self.kk_0_dict[length]
            # print(kk_0)
            out_d = torch.zeros(
                x.shape[0], self.filter_2.shape[0], self.n - length + 1, device=x.device
            )
            # print(out_d.shape)
            for kk in kk_0:
                dp_1 = kk
                dp_2 = length - kk
                full_filter = torch.zeros(
                    self.filter_2.shape[0],
                    self.filter_2.shape[1],
                    length * self.n_span,
                    device=self.filter_2.device,
                )  # Shape (out_channels, in_channels, 10)
                full_filter[
                    :,
                    :,
                    [dp_1 - self.min_split, self.n_span * dp_1 + dp_2 - self.min_split],
                ] = self.filter_2  # Place values at positions 0 and 5
                out_int = (
                    F.conv1d(x, full_filter, self.bias_2, stride=self.stride)
                    / (full_filter.size(1) * self.filter_2.size(2)) ** 0.5
                )
                # print(out_int.shape)
                out_d += out_int
            pairs_0 = self.pairs_0_dict[length]
            for pairs in pairs_0:

                dp_1 = pairs[0]
                dp_2 = pairs[1] - pairs[0]
                dp_3 = length - pairs[1]
                # print(dp_1,dp_2,dp_3)
                full_filter = torch.zeros(
                    self.filter_3.shape[0],
                    self.filter_3.shape[1],
                    length * self.n_span,
                    device=self.filter_3.device,
                )  # Shape (out_channels, in_channels, 10)
                full_filter[
                    :,
                    :,
                    [
                        dp_1 - self.min_split,
                        self.n_span * dp_1 + dp_2 - self.min_split,
                        self.n_span * (dp_1 + dp_2) + dp_3 - self.min_split,
                    ],
                ] = self.filter_3  # Place values at positions 0 and 5
                out_int = (
                    F.conv1d(x, full_filter, self.bias_3, stride=self.stride)
                    / (full_filter.size(1) * self.filter_3.size(2)) ** 0.5
                )
                # print(out_int.shape)
                out_d += out_int
            pad = torch.zeros(
                out_d.shape[0],
                out_d.shape[1],
                self.n - out_d.shape[2],
                device=out_d.device,
                dtype=out_d.dtype,
            )
            out_d_padded = torch.cat([out_d, pad], dim=2)
            outs.append(out_d_padded)
        # Step 1: Stack all elements in outs along a new dimension (dim=3)
        intermediate = torch.stack(
            outs, dim=3
        )  # Shape: (batch_size, out_channels, seq_len, num_filters)

        # Step 2: Reshape to interleave
        interleaved_out = intermediate.view(
            intermediate.shape[0], intermediate.shape[1], -1
        )

        return interleaved_out, sum_of_squares  # Return tuple
    


class hCNN_inside_L_4(nn.Module):
    def __init__(self, in_channels, nn_dim, out_channels, bias=False, norm="std"):
        super().__init__()
        len_seq=81
        self.hidden = nn.Sequential(
            ComputeSumOfSquares(),
            MyConv1d_ell_1(in_channels, nn_dim, bias=bias),
            TupleReLU(),  # Use custom ReLU that handles tuples
            MyConv1d_ell_2(nn_dim, nn_dim, len_seq, bias=bias),
            TupleReLU(),  # Again, use custom ReLU
            MyConv1d_ell_3(nn_dim, nn_dim,len_seq, bias=bias),
            TupleReLU(),  # Again, use custom ReLU
            MyConv1d_ell_4_last_parallel(nn_dim, nn_dim, bias=bias),
            TupleReLU(),  # Again, use custom ReLU
        )
        self.min_d_last=16
        self.readout = nn.Parameter(torch.randn(nn_dim, out_channels))
        if norm == "std":
            self.norm = nn_dim**0.5  # standard NTK scaling
        elif norm == "mf":
            self.norm = nn_dim  # mean-field scaling

    def forward(self, x):
        x, sum_of_squares = self.hidden(x)
        x = x[torch.arange(x.shape[0]), :, sum_of_squares - self.min_d_last]  # Dynamic indexing
        x = x @ self.readout / self.norm
        return x

class hCNN_inside_L_3(nn.Module):
    def __init__(self, in_channels, nn_dim, out_channels, bias=False, norm="std"):
        super().__init__()
        len_seq=27
        self.hidden = nn.Sequential(
            ComputeSumOfSquares(),
            MyConv1d_ell_1(in_channels, nn_dim, bias=bias),
            TupleReLU(),  # Use custom ReLU that handles tuples
            MyConv1d_ell_2(nn_dim, nn_dim, len_seq, bias=bias),
            TupleReLU(),  # Again, use custom ReLU
            MyConv1d_ell_3_last(nn_dim, nn_dim, bias=bias),
            TupleReLU(),  # Again, use custom ReLU
        )
        self.min_d_last=8
        self.readout = nn.Parameter(torch.randn(nn_dim, out_channels))
        if norm == "std":
            self.norm = nn_dim**0.5  # standard NTK scaling
        elif norm == "mf":
            self.norm = nn_dim  # mean-field scaling

    def forward(self, x):
        x, sum_of_squares = self.hidden(x)
        x = x[torch.arange(x.shape[0]), :, sum_of_squares - self.min_d_last]  # Dynamic indexing
        x = x @ self.readout / self.norm
        return x

class hCNN_inside_L_2(nn.Module):
    def __init__(self, in_channels, nn_dim, out_channels, bias=False, norm="std"):
        super().__init__()

        self.hidden = nn.Sequential(
            ComputeSumOfSquares(),
            MyConv1d_ell_1(in_channels, nn_dim, bias=bias),
            TupleReLU(),  # Use custom ReLU that handles tuples
            MyConv1d_ell_2_last(nn_dim, nn_dim, bias=bias),
            TupleReLU(),  # Again, use custom ReLU
        )
        self.min_d_last=4
        self.readout = nn.Parameter(torch.randn(nn_dim, out_channels))
        if norm == "std":
            self.norm = nn_dim**0.5  # standard NTK scaling
        elif norm == "mf":
            self.norm = nn_dim  # mean-field scaling

    def forward(self, x):
        x, sum_of_squares = self.hidden(x)
        x = x[torch.arange(x.shape[0]), :, sum_of_squares - self.min_d_last]  # Dynamic indexing
        x = x @ self.readout / self.norm
        return x


class hCNN_inside_L_2_tree_topologies(nn.Module):
    def __init__(self, in_channels, nn_dim, out_channels, bias=False, norm="std"):
        super().__init__()

        self.hidden = nn.Sequential(
            TensorChopping(),
            MyConv1d_ell_1(in_channels, nn_dim, bias=bias),
            TupleReLU(),  # Use custom ReLU that handles tuples
            MyConv1d_ell_2_last_tree_topologies(nn_dim, nn_dim, bias=bias),
            TupleReLU(),  # Again, use custom ReLU
        )
        self.min_d_last=4
        self.readout = nn.Parameter(torch.randn(nn_dim, out_channels))
        if norm == "std":
            self.norm = nn_dim**0.5  # standard NTK scaling
        elif norm == "mf":
            self.norm = nn_dim  # mean-field scaling

    def forward(self, x):
        x, t_type = self.hidden(x)
        x = x[torch.arange(x.shape[0]), :, t_type]  # Dynamic indexing
        x = x @ self.readout / self.norm
        return x


class MyConv1d_ell_1_single(nn.Module):
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
        self.stride = 1  # Stride should be the sum of both patch sizes (2+3)

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
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
        # Apply convolution separately on patch size 2
        out_2 = (
            F.conv1d(x, self.filter_2, self.bias_2, stride=self.stride)
            / (self.filter_2.size(1) * self.filter_2.size(2)) ** 0.5
        )

        # Apply convolution separately on patch size 3
        out_3 = (
            F.conv1d(x, self.filter_3, self.bias_3, stride=self.stride)
            / (self.filter_3.size(1) * self.filter_3.size(2)) ** 0.5
        )

        # Pad out_3 with one zero so that it has the same length as out_2
        pad_2 = torch.zeros(
            out_2.shape[0], out_2.shape[1], 1, device=out_2.device, dtype=out_2.dtype
        )
        out_2_padded = torch.cat([out_2, pad_2], dim=2)
        pad_3 = torch.zeros(
            out_3.shape[0], out_3.shape[1], 2, device=out_3.device, dtype=out_3.dtype
        )
        out_3_padded = torch.cat([out_3, pad_3], dim=2)

        # Interleave the outputs
        interleaved_out = torch.stack((out_2_padded, out_3_padded), dim=3).view(
            out_2.shape[0], out_2.shape[1], -1
        )

        return interleaved_out


# Define the custom 1D convolutional layer
class MyConv1d_ell_2_single(nn.Module):
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

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        l = 2
        self.n = 9
        self.min_d = 2**l
        self.max_d = 3 ** (l)
        self.min_split = 2 ** (l - 1)
        self.max_split = 3 ** (l - 1)
        self.n_span = 3 ** (l - 1) - 2 ** (l - 1) + 1
        # Dictionary to store kk_0 lists
        self.stride = self.n_span

        self.kk_0_dict = {}
        self.pairs_0_dict = {}

        for length in range(self.min_d, self.max_d + 1):  # Length of the span
            kk_0 = np.array(range(self.min_split, length - self.min_split + 1))
            kk_0 = kk_0[(kk_0 <= self.max_split) & (kk_0 >= (length - self.max_split))]
            self.kk_0_dict[length] = kk_0.tolist()
            pairs_0 = nested_ranges_as_tensor(0, length, self.min_split)
            condition_1 = pairs_0[:, 0] <= self.max_split
            condition_2 = ((pairs_0[:, 1] - pairs_0[:, 0]) <= self.max_split) & (
                (pairs_0[:, 1] - pairs_0[:, 0]) >= self.min_split
            )
            condition_3 = (pairs_0[:, 1] >= (length - self.max_split)) & (
                length - pairs_0[:, 1] >= self.min_split
            )
            # Combine all conditions
            pairs_0 = pairs_0[condition_1 & condition_2 & condition_3]
            self.pairs_0_dict[length] = pairs_0

        # Pruning condition: keep elements <= max_split and >= (length - max_split)

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, x):
        # Apply convolution separately on patch size 2
        outs = []
        for length in range(self.min_d, self.max_d + 1):
            kk_0 = self.kk_0_dict[length]
            # print(kk_0)
            out_d = torch.zeros(
                x.shape[0], self.filter_2.shape[0], self.n - length + 1, device=x.device
            )
            # print(out_d.shape)
            for kk in kk_0:
                dp_1 = kk
                dp_2 = length - kk
                full_filter = torch.zeros(
                    self.filter_2.shape[0],
                    self.filter_2.shape[1],
                    length * self.n_span,
                    device=self.filter_2.device,
                )  # Shape (out_channels, in_channels, 10)
                full_filter[
                    :,
                    :,
                    [dp_1 - self.min_split, self.n_span * dp_1 + dp_2 - self.min_split],
                ] = self.filter_2  # Place values at positions 0 and 5
                out_int = (
                    F.conv1d(x, full_filter, self.bias_2, stride=self.stride)
                    / (full_filter.size(1) * self.filter_2.size(2)) ** 0.5
                )
                # print(out_int.shape)
                out_d += out_int
            pairs_0 = self.pairs_0_dict[length]
            for pairs in pairs_0:

                dp_1 = pairs[0]
                dp_2 = pairs[1] - pairs[0]
                dp_3 = length - pairs[1]
                # print(dp_1,dp_2,dp_3)
                full_filter = torch.zeros(
                    self.filter_3.shape[0],
                    self.filter_3.shape[1],
                    length * self.n_span,
                    device=self.filter_3.device,
                )  # Shape (out_channels, in_channels, 10)
                full_filter[
                    :,
                    :,
                    [
                        dp_1 - self.min_split,
                        self.n_span * dp_1 + dp_2 - self.min_split,
                        self.n_span * (dp_1 + dp_2) + dp_3 - self.min_split,
                    ],
                ] = self.filter_3  # Place values at positions 0 and 5
                out_int = (
                    F.conv1d(x, full_filter, self.bias_3, stride=self.stride)
                    / (full_filter.size(1) * self.filter_3.size(2)) ** 0.5
                )
                # print(out_int.shape)
                out_d += out_int
            pad = torch.zeros(
                out_d.shape[0],
                out_d.shape[1],
                self.n - out_d.shape[2],
                device=out_d.device,
                dtype=out_d.dtype,
            )
            out_d_padded = torch.cat([out_d, pad], dim=2)
            outs.append(out_d_padded)
        # Step 1: Stack all elements in outs along a new dimension (dim=3)
        intermediate = torch.stack(
            outs, dim=3
        )  # Shape: (batch_size, out_channels, seq_len, num_filters)

        # Step 2: Reshape to interleave
        interleaved_out = intermediate.view(
            intermediate.shape[0], intermediate.shape[1], -1
        )

        return interleaved_out


class hCNN_inside_single(nn.Module):
    def __init__(
        self,
        in_channels,
        nn_dim,
        out_channels,
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

        self.hidden = nn.Sequential(
            MyConv1d_ell_1_single(in_channels, nn_dim, bias=bias),
            nn.ReLU(),
            MyConv1d_ell_2_single(nn_dim, nn_dim, bias=bias),
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









class MyConv1d_ell_2_last_linear(nn.Module):
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

        # Two separate filters
        self.filter_2 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_2)
        )
        self.filter_3 = nn.Parameter(
            torch.randn(out_channels, in_channels, self.filter_size_3)
        )

        l = 2
        self.n = 9
        self.min_d = 2**l
        self.max_d = 3 ** (l)
        self.min_split = 2 ** (l - 1)
        self.max_split = 3 ** (l - 1)
        self.n_span = 3 ** (l - 1) - 2 ** (l - 1) + 1
        # Dictionary to store kk_0 lists
        #self.stride = self.n_span
        #self.stride=1000

        self.kk_0_dict = {}
        self.pairs_0_dict = {}

        for length in range(self.min_d, self.max_d + 1):  # Length of the span
            kk_0 = np.array(range(self.min_split, length - self.min_split + 1))
            kk_0 = kk_0[(kk_0 <= self.max_split) & (kk_0 >= (length - self.max_split))]
            self.kk_0_dict[length] = kk_0.tolist()
            pairs_0 = nested_ranges_as_tensor(0, length, self.min_split)
            condition_1 = pairs_0[:, 0] <= self.max_split
            condition_2 = ((pairs_0[:, 1] - pairs_0[:, 0]) <= self.max_split) & (
                (pairs_0[:, 1] - pairs_0[:, 0]) >= self.min_split
            )
            condition_3 = (pairs_0[:, 1] >= (length - self.max_split)) & (
                length - pairs_0[:, 1] >= self.min_split
            )
            # Combine all conditions
            pairs_0 = pairs_0[condition_1 & condition_2 & condition_3]
            self.pairs_0_dict[length] = pairs_0

        # Pruning condition: keep elements <= max_split and >= (length - max_split)

        # Bias terms
        if bias:
            self.bias_2 = nn.Parameter(torch.randn(out_channels))
            self.bias_3 = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias_2", None)
            self.register_parameter("bias_3", None)

    def forward(self, inputs):
        x, sum_of_squares = inputs  # Unpack tuple
        # Apply convolution separately on patch size 2
        outs = []
        for length in range(self.min_d, self.max_d + 1):
            kk_0 = self.kk_0_dict[length]
            # print(kk_0)
            out_d = torch.zeros(
                x.shape[0], self.filter_2.shape[0], 1, device=x.device
            )
            # print(out_d.shape)
            for kk in kk_0:
                # Assume: x is [batch_size, in_channels, total_seq_len]

                # For a given length and kk
                dp_1 = kk
                dp_2 = length - kk

                # We need to gather two points:
                # - One at position dp1
                # - One at position dp1 + dp2

                # Suppose we know where in x to sample (indexes).
                # Gather x at these two locations.
                patch1 = x[:, :, dp_1-self.min_split]
                patch2 = x[:, :, self.n_span * dp_1 + dp_2 - self.min_split]

                # Stack or concatenate them
                patch = torch.stack([patch1, patch2], dim=2)  # Shape: [batch_size, in_channels, 2]

                # Now apply the 1x2 convolution manually:
                # Shape of filter_2: [out_channels, in_channels, 2]
                out_d += torch.einsum('bic,oic->bo', patch, self.filter_2) / (self.filter_2.size(1) * self.filter_2.size(2))**0.5

                # Possibly normalize
            pairs_0 = self.pairs_0_dict[length]
            for pairs in pairs_0:

                dp_1 = pairs[0]
                dp_2 = pairs[1] - pairs[0]
                dp_3 = length - pairs[1]
                # print(dp_1,dp_2,dp_3)


                patch1 = x[:, :, dp_1 - self.min_split]
                patch2 = x[:, :, self.n_span * dp_1 + dp_2 - self.min_split]
                patch3 = x[:, :, self.n_span * (dp_1 + dp_2) + dp_3 - self.min_split]

                # Stack or concatenate them
                patch = torch.stack([patch1, patch2,patch3], dim=3)  # Shape: [batch_size, in_channels, 3]

                # Now apply the 1x2 convolution manually:
                # Shape of filter_2: [out_channels, in_channels, 2]
                out_d += torch.einsum('bic,oic->bo', patch, self.filter_3) / (self.filter_3.size(1) * self.filter_3.size(2))**0.5
                
        
            outs.append(out_d)
        # Step 1: Stack all elements in outs along a new dimension (dim=3)
        intermediate = torch.stack(
            outs, dim=3
        )  # Shape: (batch_size, out_channels, seq_len, num_filters)

        # Step 2: Reshape to interleave
        interleaved_out = intermediate.view(
            intermediate.shape[0], intermediate.shape[1], -1
        )

        return interleaved_out, sum_of_squares  # Return tuple
    