import math

import torch
import torch.nn as nn


class Vgg(nn.Module):
    def __init__(self, fc_layer=512, classes=10):
        super(Vgg, self).__init__()
        """ Initialize VGG simplified Module
        Args: 
            fc_layer: input feature number for the last fully MLP block
            classes: number of image classes
        """
        self.fc_layer = fc_layer
        self.classes = classes

        # todo: construct the simplified VGG network blocks
        # input shape: [bs, 3, 32, 32]
        # layers and output feature shape for each block:
        # # conv_block1 (Conv2d, ReLU, MaxPool2d) --> [bs, 64, 16, 16]
        # # conv_block2 (Conv2d, ReLU, MaxPool2d) --> [bs, 128, 8, 8]
        # # conv_block3 (Conv2d, ReLU, MaxPool2d) --> [bs, 256, 4, 4]
        # # conv_block4 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 2, 2]
        # # conv_block5 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 1, 1]
        # # classifier (Linear, ReLU, Dropout2d, Linear) --> [bs, 10] (final output)

        # hint: stack layers in each block with nn.Sequential, e.x.:
        # # self.conv_block1 = nn.Sequential(
        # #     layer1,
        # #     layer2,
        # #     layer3,
        # #     ...)
        # for all conv layers, set: kernel=3, padding=1

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv_block2 =  nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_block = nn.Sequential(
            nn.Linear(512, fc_layer),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_layer, classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        """
        :param x: input image batch tensor, [bs, 3, 32, 32]
        :return: score: predicted score for each class (10 classes in total), [bs, 10]
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = x.flatten(start_dim=1)
        score = self.fc_block(x)

        return score

