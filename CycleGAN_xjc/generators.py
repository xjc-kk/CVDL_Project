'''
This file defines the Generator network with 9 resnet blocks.
'''
import torch
from torch import nn
import functools
from network import conv_norm_relu, dconv_norm_relu, ResidualBlock, get_norm_layer, init_network


class Generator(nn.Module):
    '''
    This generator contains 9 resnet blocks.
    '''
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True, use_bias = False):
        super(Generator, self).__init__()
        
        self.pad = nn.ReflectionPad2d(3)
        self.layer1 = conv_norm_relu(input_nc, ngf, kernel_size=7, norm_layer=norm_layer, bias=use_bias)
        self.layer2 = conv_norm_relu(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, bias=use_bias)
        self.layer3 = conv_norm_relu(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, norm_layer=norm_layer, bias=use_bias)

        self.res1 = ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)
        self.res2 = ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)
        self.res3 = ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)
        self.res4 = ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)
        self.res5 = ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)
        self.res6 = ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)
        self.res7 = ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)
        self.res8 = ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)
        self.res9 = ResidualBlock(ngf * 4, norm_layer, use_dropout, use_bias)

        self.layer4 = dconv_norm_relu(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, norm_layer=norm_layer, bias=use_bias)
        self.layer5 = dconv_norm_relu(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, norm_layer=norm_layer, bias=use_bias)
        self.conv6 = nn.Conv2d(ngf, output_nc, kernel_size=7)
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        Conv Layer -- Residual Block -- Deconv Layer
        '''
        out = self.pad(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.res7(out)
        out = self.res8(out)
        out = self.res9(out)

        out = self.layer4(out)
        out = self.layer5(out)
        out = self.pad(out)
        out = self.conv6(out)
        out = self.tanh(out)

        return out


def define_Gen(input_nc, output_nc, ngf, norm='batch', use_dropout=False, gpu_ids=[0]):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    gen_net = Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, use_bias = use_bias)

    return init_network(gen_net, gpu_ids)