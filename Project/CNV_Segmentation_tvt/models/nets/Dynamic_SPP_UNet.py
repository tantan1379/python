# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019
@author: Fsl
"""

import torch
from torchvision import models
import torch.nn as nn
#from models.nets.resnet import resnet34

from torch.nn import functional as F
import torchsummary
from torch.nn import init
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class Dynamic_SPP_UNet(nn.Module):
    def __init__(self,in_channels=1,num_classes=4,ccm=True,
                 norm_layer=nn.BatchNorm2d,is_training=True,expansion=2,base_channel=32):
        super(Dynamic_SPP_UNet,self).__init__()
        
        self.backbone =models.resnet34(pretrained=False)
        self.expansion=expansion
        self.base_channel=base_channel
        self.expan=[64,64,128,256,512]

        self.is_training = is_training



        self.spp1 = SPPblock(self.expan[0],self.expan[1])
        self.spp2 = SPPblock(self.expan[1],self.expan[2])
        self.spp3 = SPPblock(self.expan[2],self.expan[3])

        self.decoder5=DecoderBlock(self.expan[-1],self.expan[-2],relu=False,last=True) #256
        self.decoder4=DecoderBlock(self.expan[-2],self.expan[-3],relu=False) #128
        self.decoder3=DecoderBlock(self.expan[-3],self.expan[-4],relu=False) #64
        self.decoder2=DecoderBlock(self.expan[-4],self.expan[-5]) #32

      

        self.main_head= BaseNetHead(self.expan[0], num_classes, 2,
                             is_aux=False, norm_layer=norm_layer)
       
        self.relu = nn.ReLU()

        
    
    def forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)#1/2  64
        
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)#1/4   64
        c3 = self.backbone.layer2(c2)#1/8   128
        c4 = self.backbone.layer3(c3)#1/16   256
        c5 = self.backbone.layer4(c4)#1/32   512

        c2_sp = self.spp1(c1,c2)
        c3_sp = self.spp2(c2,c3)
        c4_sp = self.spp3(c3,c4)



        d4=self.relu(self.decoder5(c5)+c4_sp)  #256
        d3=self.relu(self.decoder4(d4)+c3_sp)  #128
        d2=self.relu(self.decoder3(d3)+c2_sp) #64
        d1=self.decoder2(d2)+c1 #32
        main_out=self.main_head(d1)
        main_out=torch.sigmoid(main_out)

            
        return main_out
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)


class SPPblock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(SPPblock, self).__init__()
        self.Params = nn.Parameter(torch.ones(4))

        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=[4, 4], stride=4)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
        self.pool5 = nn.MaxPool2d(kernel_size=[6, 8], stride=8)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,bias=False),
        )
        self.conv_smoonth = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,kernel_size=1,bias=False),
            nn.ReLU(inplace=True)
                )

    def forward(self, x_L,x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.layer2 = F.upsample(self.conv(self.pool2(x_L)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x_L)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x_L)), size=(h, w), mode='bilinear')
        self.layer5 = F.upsample(self.conv(self.pool5(x_L)), size=(h, w), mode='bilinear')

        out = x+self.Params[0]*self.layer2+self.Params[1]*self.layer3+self.Params[2]*self.layer4+self.Params[3]*self.layer5
        out1 = self.conv_smoonth(out)

        return out1


class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3=nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3=nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False))
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1_2 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1_2 = nn.Conv2d(32, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)
        fm = self.conv_1x1_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1_2(fm)
        return output
class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d,scale=2,relu=True,last=False):
        super(DecoderBlock, self).__init__()
       

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
       
        self.scale=scale
        self.last=last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last==False:
            x = self.conv_3x3(x)
        if self.scale>1:
            x=F.interpolate(x,scale_factor=self.scale,mode='bilinear',align_corners=True)
        x=self.conv_1x1(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x
    

if __name__ == '__main__':
    images = torch.rand(1, 3, 256, 256).cuda(0)
    model = Dynamic_SPP_UNet(num_classes=2)
    import numpy as np
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    model = model.cuda(0)
    print(model(images)[0].size())