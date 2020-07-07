'''
This file defines the Discriminator network.
'''
import torch
from torch import nn
from torch.nn import functional as F
import functools
from network import conv_norm_lrelu, get_norm_layer, init_network


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, True)
        self.layer2 = conv_norm_lrelu(ndf, ndf * 2, kernel_size=4, stride=2, norm_layer= norm_layer, padding=1, bias=use_bias)
        self.layer3 = conv_norm_lrelu(ndf * 2, ndf * 4, kernel_size=4, stride=2, norm_layer= norm_layer, padding=1, bias=use_bias)
        self.layer4 = conv_norm_lrelu(ndf * 4, ndf * 8, kernel_size=4, stride=2, norm_layer= norm_layer, padding=1, bias=use_bias)
        self.conv5 = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lrelu1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv5(out)

        return out
        

def define_Dis(input_nc, ndf, norm='batch', gpu_ids=[0]):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    dis_net = Discriminator(input_nc, ndf, norm_layer=norm_layer, use_bias=use_bias)

    return init_network(dis_net, gpu_ids)