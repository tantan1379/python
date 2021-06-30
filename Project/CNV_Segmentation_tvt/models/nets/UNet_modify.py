import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.layers import unetConv2, unetUp, unetConv2_dilation
from models.utils.init_weights import init_weights
class UNet_m(nn.Module):

    def __init__(self, in_channels=1,n_classes=4,feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_m, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale


        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.squeeze4 = nn.Sequential(nn.Conv2d(filters[4], filters[3], kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(filters[3]), nn.ReLU(inplace=True))
        self.bnre4=nn.Sequential(nn.BatchNorm2d(filters[3]), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(filters[3], filters[2], kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(filters[2]), nn.ReLU(inplace=True))
        self.bnre3 = nn.Sequential(nn.BatchNorm2d(filters[2]), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(filters[2], filters[1], kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(filters[1]), nn.ReLU(inplace=True))
        self.bnre2 = nn.Sequential(nn.BatchNorm2d(filters[1]), nn.ReLU(inplace=True))
        self.squeeze1 = nn.Sequential(nn.Conv2d(filters[1], filters[0], kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True))
        self.bnre1 = nn.Sequential(nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True))

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)       # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)     # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)     # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)     # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)   # 256*32*64
        #print(center.shape)   #torch.Size([4, 256, 16, 16])

        md4 = F.interpolate(center, conv4.shape[2:],  mode='bilinear')
        #print(md4.shape)      #torch.Size([4, 256, 32, 32])
        md4 = self.squeeze4(md4)
        mul4 = torch.mul(md4,conv4)
        result4 = self.bnre4(torch.add(mul4,conv4))
        print(result4.shape)    #torch.Size([4, 128, 32, 32])

        md3 = F.interpolate(result4, conv3.shape[2:],  mode='bilinear')
        print(md3.shape)           #torch.Size([4, 128, 64, 64])
        md3 = self.squeeze3(md3)
        mul3= torch.mul(md3,conv3)
        result3 = self.bnre3(torch.add(mul3,conv3))

        md2 = F.interpolate(result3, conv2.shape[2:],  mode='bilinear')
        md2 = self.squeeze2(md2)
        mul2 = torch.mul(md2,conv2)
        result2 = self.bnre2(torch.add(mul2,conv2))

        md1 = F.interpolate(result2, conv1.shape[2:],  mode='bilinear')
        md1 = self.squeeze1(md1)
        mul1 = torch.mul(md1,conv1)
        result1 = self.bnre1(torch.add(mul1,conv1))



        up4 = self.up_concat4(center,result4)  # 128*64*128
        up3 = self.up_concat3(up4,result3)     # 64*128*256
        up2 = self.up_concat2(up3,result2)     # 32*256*512
        up1 = self.up_concat1(up2,result1)     # 16*512*1024

        final_1 = self.final_1(up1)


        return torch.sigmoid(final_1)




# ------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = UNet_m(in_channels=3, n_classes=1, is_deconv=True).cuda()
    #print(net)
    x = torch.rand((4, 3, 256, 256)).cuda()
    forward = net.forward(x)
    import numpy as np
    # print(forward)
#     # print(type(forward))
    model = UNet_m(in_channels=3, n_classes=1, is_deconv=True).cuda()
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    tmp = torch.randn(2, 3, 256, 256).cuda()
    y = torch.randn(1, 448, 448)

    import time

    start_time = time.time()
    print(model(tmp).shape)
    end_time = time.time()
    print("Time ==== {}".format(end_time - start_time))
    print('done')
#
