import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from lib.backbones import dla
from lib.backbones.dlaup import DLAUp
from lib.backbones.hourglass import get_large_hourglass_net
from lib.backbones.hourglass import load_pretrian_model
from lib.backbones.pose_resnet import get_pose_net


class CenterNet3D(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', num_class=3, downsample=4, cfg=None):
        """
        CenterNet for monocular 3D object detection.
        :param backbone: the backbone of pipeline, such as dla34.
        :param neck: the necks of detection, such as dla_up.
        :param downsample: the ratio of down sample. [4, 8, 16, 32]
        :param head_conv: the channels of convolution in head. default: 256
        """
        assert downsample in [4, 8, 16, 32]
        super().__init__()

        self.heads = {'heatmap': num_class, 'offset_2d': 2, 'size_2d' :2, 'depth': 2, 'offset_3d': 2, 'size_3d':3, 'heading': 24}

        self.use_dlaup = True
        if backbone == 'dla34':
            self.backbone = getattr(dla, backbone)(pretrained=True, return_levels=True)
            channels = self.backbone.channels  # channels list for feature maps generated by backbone
            self.first_level = int(np.log2(downsample))
            scales = [2 ** i for i in range(len(channels[self.first_level:]))]
            self.neck = DLAUp(channels[self.first_level:], scales_list=scales)   # feature fusion [such as DLAup, FPN]
        elif backbone == 'res18':
            print("using pose_resnet...")
            self.use_dlaup = False
            channels = [16, 32, 64, 128, 256, 512]
            self.first_level = int(np.log2(downsample))
            scales = [2 ** i for i in range(len(channels[self.first_level:]))]

            num_layers = 18
            heads = {}
            self.backbone = get_pose_net(num_layers, heads)
        else:
            raise NotImplementedError

        self.head_conv = 256

        # initialize the head of pipeline, according to heads setting.
        for head in self.heads.keys():
            output_channels = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, output_channels, kernel_size=1, stride=1, padding=0, bias=True))

            # initialization
            if 'heatmap' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def forward(self, input):
        feat = self.backbone(input)
        if self.use_dlaup:
            feat = self.neck(feat[self.first_level:])

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)

        return ret


    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




if __name__ == '__main__':
    import torch
    net = CenterNet3D(backbone='dla34')
    print(net)

    input = torch.randn(4, 3, 384, 1280)
    print(input.shape, input.dtype)
    output = net(input)
    print(output.keys())


