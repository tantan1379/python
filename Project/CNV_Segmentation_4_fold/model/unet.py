"""
Created on Sun May 30 12:42 2021

@author:twh
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=False, is_batchnorm=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x/self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels,filters[0],self.is_batchnorm)
        self.conv2 = unetConv2(filters[0],filters[1],self.is_batchnorm)
        self.conv3 = unetConv2(filters[1],filters[2],self.is_batchnorm)
        self.conv4 = unetConv2(filters[2],filters[3],self.is_batchnorm)
        self.center = unetConv2(filters[3],filters[4],self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4],filters[3],self.is_deconv)
        self.up_concat3 = unetUp(filters[3],filters[2],self.is_deconv)
        self.up_concat2 = unetUp(filters[2],filters[1],self.is_deconv)
        self.up_concat1 = unetUp(filters[1],filters[0],self.is_deconv)
        # final conv
        self.final = nn.Conv2d(filters[0],n_classes,1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                init_weights(m,'kaiming')
            elif isinstance(m,nn.BatchNorm2d):
                init_weights(m,'kaiming')
        
    def forward(self,inputs):               # 3*256*512                    
        # print("inputs:",inputs.shape)                 
        # encoder                           
        conv1 = self.conv1(inputs)          # 32*256*512
        maxpool1 = self.maxpool(conv1)      # 32*128*256
        # print("maxpool1:",maxpool1.shape)
        conv2 = self.conv2(maxpool1)        # 64*128*256
        maxpool2 = self.maxpool(conv2)      # 64*64*128
        # print("maxpool2:",maxpool2.shape)
        conv3 = self.conv3(maxpool2)        # 128*64*128
        maxpool3 = self.maxpool(conv3)      # 128*32*64
        # print("maxpool3:",maxpool3.shape)
        conv4 = self.conv4(maxpool3)        # 256*32*64
        maxpool4 = self.maxpool(conv4)      # 256*16*32
        # print("maxpool4:",maxpool4.shape)
        center = self.center(maxpool4)      # 512*16*32
        # print("center:",center.shape)       
        # decoder(转置卷积+拼接+卷积块)
        up4 = self.up_concat4(center,conv4) # 512*16*32(ConvTranspose)->256*32*64(concat)->512*32*64(conv2)->256*32*64
        # print("up4:",up4.shape)
        up3 = self.up_concat3(up4,conv3)    # 512*16*32(ConvTranspose)->128*64*128(concat)->256*64*128(conv2)->128*64*128
        # print("up3:",up3.shape)
        up2 = self.up_concat2(up3,conv2)    # 512*16*32(ConvTranspose)->64*128*256(concat)->128*128*256(conv2)->64*128*256
        # print("up2:",up2.shape)
        up1 = self.up_concat1(up2,conv1)    # 512*16*32(ConvTranspose)->32*256*512(concat)->64*256*512(conv2)->32*256*512
        # print("up1:",up1.shape)
        final = self.final(up1)             # 1*256*512
        # print("final:",final.shape)
        return final                        # 1*256*512




class unetConv2(nn.Module):
    def __init__(self,in_size,out_size,is_batchnorm,n=2,ks=3,stride=1,padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks # kernel_size
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1,n+1):
                conv = nn.Sequential(nn.Conv2d(in_size,out_size,ks,s,p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True))
                setattr(self,'conv%d'%i,conv)
                in_size = out_size
        
        else:
            for i in range(1,n+1):
                conv = nn.Sequential(nn.Conv2d(in_size,out_size,ks,s,p),
                                     nn.ReLU(inplace=True))
                setattr(self,'conv%d'%i,conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m,init_type="kaiming")
    
    def forward(self,inputs):
        x = inputs
        for i in range(1,self.n+1):
            conv = getattr(self,'conv%d'%i)
            x = conv(x)
        return x

class unetUp(nn.Module):
    def __init__(self,in_size,out_size,is_deconv,n_concat=2):
        super(unetUp,self).__init__()
        self.conv = unetConv2(in_size+(n_concat-2)*out_size,out_size,False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size,out_size,kernel_size=2,stride=2,padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size,out_size,1)
            )
        
        # initialize the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2')!=-1:
                continue
            init_weights(m,init_type='kaiming')
        
    def forward(self,high_feature,*low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0,feature],dim=1)  # dim=1是通道维
        return self.conv(outputs0)

# 权重初始化（对于激活函数ReLU一般采用何凯明提出的kaiming正态分布初始化卷积层参数）
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    model = Unet()
    x = torch.Tensor(1,3,256,512)
    output = model(x)
    # print(output)
    # print("output:",output[0].shape)
    # print(model)