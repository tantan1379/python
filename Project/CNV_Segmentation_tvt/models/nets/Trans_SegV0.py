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

class Trans_Seg_V0(nn.Module):
    def __init__(self,out_planes=4,
                 norm_layer=nn.BatchNorm2d,is_training=True,expansion=2,base_channel=32):
        super(Trans_Seg_V0,self).__init__()
        print("============ Trans_SegV0 ============")
        
        self.backbone =models.resnet18(pretrained=True)
        self.expansion=expansion
        self.base_channel=base_channel

        expan=[64,64,128,256,512]


        self.is_training = is_training

        self.decoder5=DecoderBlock(expan[-1],expan[-2],relu=False,last=True) #256
        self.decoder4=DecoderBlock(expan[-2],expan[-3],relu=False) #128
        self.decoder3=DecoderBlock(expan[-3],expan[-4],relu=False) #64
        self.decoder2=DecoderBlock(expan[-4],expan[-5]) #32

        self.trans = Multi_Trans(train_dim=expan[-1])

        self.main_head= BaseNetHead(expan[0], out_planes, 2,
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

        c5_Trans,_ = self.trans(c5,c2,c3,c4)



        d4=self.relu(self.decoder5(c5_Trans)+c4)  #256
        d3=self.relu(self.decoder4(d4)+c3)  #128
        d2=self.relu(self.decoder3(d3)+c2) #64
        d1=self.decoder2(d2)+c1 #32
        main_out=self.main_head(d1)
        main_out=torch.sigmoid(main_out)

            
        return main_out
    

class SPP_Q(nn.Module):
    def __init__(self,in_ch,out_ch,down_scale,ks=3):
        super(SPP_Q, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=1, padding=ks // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.Down = nn.Upsample(scale_factor=down_scale,mode="bilinear")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x_d = self.Down(x)
        x_out = self.Conv(x_d)
        return x_out

class Multi_Trans(nn.Module):
    """ Self attention Layer"""
    def __init__(self,train_dim,filters=[64,128,256]):
        super(Multi_Trans,self).__init__()
        self.chanel_in = train_dim

        self.SPP_Q_1 = SPP_Q(in_ch=filters[0],out_ch=train_dim,down_scale=1/8,ks=3)
        self.SPP_Q_2 = SPP_Q(in_ch=filters[1],out_ch=train_dim,down_scale=1/4,ks=3)
        self.SPP_Q_3 = SPP_Q(in_ch=filters[2],out_ch=train_dim,down_scale=1/2,ks=3)


        self.query_conv = nn.Conv2d(in_channels = train_dim , out_channels = train_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = train_dim , out_channels = train_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = train_dim , out_channels = train_dim , kernel_size= 1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x,x_c1,x_c2,x_c3):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        Multi_X = self.SPP_Q_1(x_c1)+self.SPP_Q_2(x_c2)+self.SPP_Q_3(x_c3)
        proj_query = self.query_conv(Multi_X).view(m_batchsize,-1,width*height).permute(0,2,1) # B*N*C


        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B*C*N

        energy = torch.bmm(proj_query,proj_key) # batch的matmul B*N*N
        attention = self.softmax(energy) # B * (N) * (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1, width*height) # B * C * N

        out = torch.bmm(proj_value,attention.permute(0,2,1) ) # B*C*N
        out = out.view(m_batchsize,C,width,height) # B*C*H*W

        out = self.gamma*out + x
        return out,attention

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
            # x=self.sap(x)
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
    
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs

    



if __name__ == '__main__':
    images = torch.rand(2, 3, 224, 224).cuda(0)
    model = Trans_Seg_V0(out_planes=1)
    import numpy as np
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    model = model.cuda(0)
    print(model(images)[0].size())
