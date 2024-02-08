from __future__ import absolute_import, division, print_function

import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SegNetLite(nn.Module):

    def __init__(self, kernel_sizes=[3, 3, 3, 3], down_filter_sizes=[32, 64, 128, 256],
            up_filter_sizes=[128, 64, 32, 32], conv_paddings=[1, 1, 1, 1],
            pooling_kernel_sizes=[2, 2, 2, 2], pooling_strides=[2, 2, 2, 2], **kwargs):
        """Initialize SegNet Module

        Args:
            kernel_sizes (list of ints): kernel sizes for each convolutional layer in downsample/upsample path.
            down_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the downsample path.
            up_filter_sizes (list of ints): number of filters (out channels) of each convolutional layer in the upsample path.
            conv_paddings (list of ints): paddings for each convolutional layer in downsample/upsample path.
            pooling_kernel_sizes (list of ints): kernel sizes for each max-pooling layer and its max-unpooling layer.
            pooling_strides (list of ints): strides for each max-pooling layer and its max-unpooling layer.
        """
        super(SegNetLite, self).__init__()
        self.num_down_layers = len(kernel_sizes)
        self.num_up_layers = len(kernel_sizes)

        input_size = 3 # initial number of input channels
        # Construct downsampling layers.
        # As mentioned in the assignment, blocks of the downsampling path should have the
        # following output dimension (igoring batch dimension):
        # 3 x 64 x 64 (input) -> 32 x 32 x 32 -> 64 x 16 x 16 -> 128 x 8 x 8 -> 256 x 4 x 4
        # each block should consist of: Conv2d->BatchNorm2d->ReLU->MaxPool2d
        layers_conv_down = [nn.Conv2d(input_size if i == 0 else down_filter_sizes[i-1],
                                        down_filter_sizes[i],
                                        kernel_sizes[i], 
                                        padding=conv_paddings[i]) for i in range(self.num_down_layers)]
        layers_bn_down = [nn.BatchNorm2d(down_filter_sizes[i]) for i in range(self.num_down_layers)]
        layers_pooling = [nn.MaxPool2d(pooling_kernel_sizes[i], 
                                       pooling_strides[i],
                                       return_indices=True) for i in range(self.num_down_layers)]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # package can track gradients and update parameters of these layers
        self.layers_conv_down = nn.ModuleList(layers_conv_down)
        self.layers_bn_down = nn.ModuleList(layers_bn_down)
        self.layers_pooling = nn.ModuleList(layers_pooling)

        # Construct upsampling layers
        # As mentioned in the assignment, blocks of the upsampling path should have the
        # following output dimension (igoring batch dimension):
        # 256 x 4 x 4 (input) -> 128 x 8 x 8 -> 64 x 16 x 16 -> 32 x 32 x 32 -> 32 x 64 x 64
        # each block should consist of: MaxUnpool2d->Conv2d->BatchNorm2d->ReLU
        layers_conv_up = [nn.Conv2d(down_filter_sizes[-1] if i == 0 else up_filter_sizes[i-1],
                                        up_filter_sizes[i],
                                        kernel_sizes[i],
                                        padding=conv_paddings[i]) for i in range(self.num_up_layers)]
        layers_bn_up = [nn.BatchNorm2d(up_filter_sizes[i]) for i in range(self.num_up_layers)]
        layers_unpooling = [nn.MaxUnpool2d(pooling_kernel_sizes[i],
                                             pooling_strides[i]) for i in range(self.num_up_layers)]

        # Convert Python list to nn.ModuleList, so that PyTorch's autograd
        # can track gradients and update parameters of these layers
        self.layers_conv_up = nn.ModuleList(layers_conv_up)
        self.layers_bn_up = nn.ModuleList(layers_bn_up)
        self.layers_unpooling = nn.ModuleList(layers_unpooling)

        self.relu = nn.ReLU(True)

        # Implement a final 1x1 convolution to to get the logits of 11 classes (background + 10 digits)
        self.conv_final = nn.Conv2d(up_filter_sizes[-1], 11, 1)

    def forward(self, x):

        indices = []
    
        for i in range(len(self.layers_conv_down)):
            x = self.layers_conv_down[i](x)
            x = self.layers_bn_down[i](x)
            x = self.relu(x)
            x, index = self.layers_pooling[i](x)

            indices.append(index)
        
        for i in range(len(self.layers_conv_up)):
            x = self.layers_unpooling[i](x, indices[-(i+1)])
            x = self.layers_conv_up[i](x)
            x = self.layers_bn_up[i](x)
            x = self.relu(x)
        
        x = self.conv_final(x)
        return x


def get_seg_net(**kwargs):

    model = SegNetLite(**kwargs)

    return model
