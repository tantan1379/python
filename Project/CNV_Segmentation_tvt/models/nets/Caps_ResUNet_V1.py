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
from torch.nn import init
try:
    from Capsules import convolutionalCapsule
except:
    from .Capsules import convolutionalCapsule

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class Caps_ResUNet_V1(nn.Module):
    def __init__(self,in_channels=1,num_classes=4,ccm=True,
                 norm_layer=nn.BatchNorm2d,is_training=True,expansion=2,base_channel=32):
        super(Caps_ResUNet_V1,self).__init__()
        
        self.backbone =models.resnet34(pretrained=False)
        self.expansion=expansion
        self.base_channel=base_channel
        self.expan=[64,64,128,256,512]

        self.is_training = is_training

        self.Caps = convolutionalCapsule(
            in_capsules=32, out_capsules=32, in_channels=16, out_channels=16, num_routes=3, batch_norm=False
        )

        self.decoder5=DecoderBlock(self.expan[-1],self.expan[-2],relu=False,last=True) #256
        self.decoder4=DecoderBlock(self.expan[-2],self.expan[-3],relu=False) #128
        self.decoder3=DecoderBlock(self.expan[-3],self.expan[-4],relu=False) #64
        self.decoder2=DecoderBlock(self.expan[-4],self.expan[-5]) #32

      

        self.main_head= BaseNetHead(self.expan[0], num_classes, 2,
                             is_aux=False, norm_layer=norm_layer)
        self.bn_c5 = nn.BatchNorm2d(self.expan[-1])

        self.relu = nn.ReLU()

        
    
    def forward(self, x):
        batch_size = x.size(0)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)#1/2  64
        
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)#1/4   64
        c3 = self.backbone.layer2(c2)#1/8   128
        c4 = self.backbone.layer3(c3)#1/16   256
        c5 = self.backbone.layer4(c4)#1/32   512


        c5 = self.bn_c5(c5)
        # print(c5)
        c_5_reshape = c5.view(batch_size,32,16,c5.size(-2),c5.size(-1))
        c5_caps = self.Caps(c_5_reshape)
        # print('c5_caps',c5_caps)
        # c5_caps=self.bn_c5(c5_caps)
        # print('=====================')
        # print('c5_caps', c5_caps)
        c5_caps_reshape = c5_caps.view(batch_size,self.expan[-1],c5.size(-2),c5.size(-1))
        c5_caps_reshape =self.bn_c5(c5_caps_reshape)
        print('c5_caps_reshape',c5_caps_reshape)


        d4=self.relu(self.decoder5(c5_caps_reshape)+c4)  #256
        d3=self.relu(self.decoder4(d4)+c3)  #128
        d2=self.relu(self.decoder3(d3)+c2) #64
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
    model = Caps_ResUNet_V1(num_classes=2)
    import numpy as np
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    model = model.cuda(0)
    print(model(images)[0].size())