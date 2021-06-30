import torch.nn as nn
import math
import torch
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import torchvision


try:
    from deformable_conv import DeformConv2d
except:
    from .deformable_conv import DeformConv2d

__all__ = ['ResNet', 'resnet18', 'resnet34']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return scale


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y = self.conv1x1(torch.cat([y_avg, y_max], 1)).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)


class Dilat_Chnnel_Atten(nn.Module):
    def __init__(self, inplanes):
        super(Dilat_Chnnel_Atten, self).__init__()
        self.dilate1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, padding=1, dilation=1, bias=False),
            # nn.ReLU(inplace=True),
        )

        self.dilate2 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, padding=3, dilation=3, bias=False),
            # nn.ReLU(inplace=True),
        )
        self.dilate3 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, padding=3, dilation=3, bias=False),
            # nn.ReLU(inplace=True),
        )
        self.dilate4 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, padding=5, dilation=5, bias=False),
            # nn.ReLU(inplace=True),
        )
        self.channel_att = SELayer(channel=inplanes)

        self.smooth = nn.Sequential(
            # DeformConv2d(inplanes, inplanes, kernel_size=3, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)
        self.spatial = SpatialGate()
        self.gamma = torch.nn.Parameter(torch.Tensor([1.0]))
        self.beta = torch.nn.Parameter(torch.Tensor([1.0]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x):
        residual = x
        x_1 = self.dilate1(x)
        x_2 = self.dilate2(x)
        x_3 = self.dilate3(x)
        x_4 = self.dilate4(x)

        x_att_in = self.smooth(torch.cat([x_1, x_2, x_3, x_4], dim=1))
        x_CH_Att = x_att_in * (self.channel_att(x_att_in))
        x_SP_Att = x_att_in * (self.spatial(x_att_in))

        residual = self.relu(residual + self.conv1x1(x_att_in + self.gamma * x_CH_Att + self.beta * x_SP_Att))

        return residual


class Skip_guidance_HR(nn.Module):
    def __init__(self, inplanes):
        super(Skip_guidance_HR, self).__init__()

        self.smooth = nn.Sequential(
            nn.Conv2d(2 * inplanes, inplanes, kernel_size=1, bias=False),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

        self.spatial = SpatialGate()
        self.channel_att = SELayer(channel=inplanes)

        self.gamma = torch.nn.Parameter(torch.Tensor([1.0]))
        self.beta = torch.nn.Parameter(torch.Tensor([1.0]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x, x_down):
        residual = x
        x_att_in = self.smooth(torch.cat([x, x_down], 1))
        sp_att = x_att_in * (self.spatial(x_att_in))
        ch_att = x_att_in * (self.channel_att(x_att_in))

        return self.relu(residual + self.conv1x1(x_att_in + self.gamma * sp_att + self.beta * ch_att))


class Skip_guidance(nn.Module):
    def __init__(self, inplanes):
        super(Skip_guidance, self).__init__()

        self.smooth = nn.Sequential(
            nn.Conv2d(3 * inplanes, inplanes, kernel_size=1, bias=False),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)
        self.spatial = SpatialGate()
        self.channel_att = SELayer(channel=inplanes)
        self.gamma = torch.nn.Parameter(torch.Tensor([1.0]))
        self.beta = torch.nn.Parameter(torch.Tensor([1.0]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x, x_up, x_down):
        residual = x
        x_att_in = self.smooth(torch.cat([x, x_down, x_up], 1))
        sp_att = x_att_in * (self.spatial(x_att_in))
        ch_att = x_att_in * (self.channel_att(x_att_in))
        return self.relu(residual + self.conv1x1(x_att_in + self.gamma * sp_att + self.beta * ch_att))





class transpose_up_add(nn.Module):
    def __init__(self, inplanes, conv):
        super(transpose_up_add, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)


        self.conv_2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)

        self.conv_1.weight = conv.conv1.weight
        self.bn1.weight = conv.bn1.weight
        self.conv_2.weight = conv.conv2.weight
        self.bn2.weight = conv.bn2.weight


    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv_1(x)))
        x = self.relu(self.bn2(self.conv_2(x)))
        return self.relu(residual+x)



class ResNet(nn.Module):

    def __init__(self, num_classes=1):
        super(ResNet, self).__init__()
        self.baseline = torchvision.models.resnet34(pretrained=True)
        filters = [64, 64, 128, 256, 512]
        self.conv1 = self.baseline.conv1
        self.bn1 = self.baseline.bn1
        self.relu = self.baseline.relu
        self.layer1 = nn.Sequential(
            self.baseline.maxpool,
            self.baseline.layer1
        )
        self.layer2 = self.baseline.layer2
        self.layer3 = self.baseline.layer3
        self.layer4 = self.baseline.layer4

        self.Conv_HR = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )


        self.transpose_4 = transpose_up_add(filters[3],self.baseline.layer3[1])
        self.transpose_3 = transpose_up_add(filters[2],self.baseline.layer2[1])
        self.transpose_2 = transpose_up_add(filters[1],self.baseline.layer1[1])
        self.transpose_1 = transpose_up_add(filters[0],self.baseline.layer1[1])
        self.transpose_0 = transpose_up_add(filters[0],self.baseline.layer1[1])

        self.conv1x1_4 = nn.Conv2d(filters[4], filters[3], kernel_size=1, bias=False)
        self.conv1x1_3 = nn.Conv2d(filters[3], filters[2], kernel_size=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(filters[2], filters[1], kernel_size=1, bias=False)
        self.conv1x1_1 = nn.Conv2d(filters[1], filters[0], kernel_size=1, bias=False)
        self.conv1x1_0 = nn.Conv2d(filters[0], filters[0], kernel_size=1, bias=False)

        self.down_conv_HR = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )



        self.down_conv_C0 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )


        self.down_conv_C1 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )


        self.down_conv_C2 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True)
        )



        self.down_conv_C3 = nn.Sequential(
            nn.Conv2d(filters[3], filters[4], kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True)
        )


        self.upSample_C4 = nn.Sequential(
            nn.ConvTranspose2d(filters[4], filters[3], 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True)
        )

        self.upSample_C3 = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[2], 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )

        self.upSample_C2 = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )
        self.upSample_C1 = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[0], 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        self.upSample_C0 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], filters[0], 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )

        self.down_apha_HR = torch.nn.Parameter(torch.Tensor([1.0]))
        self.down_apha_C0 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.down_apha_C1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.down_apha_C2 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.down_apha_C3 = torch.nn.Parameter(torch.Tensor([1.0]))

        self.up_apha_C4 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.up_apha_C3 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.up_apha_C2 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.up_apha_C1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.up_apha_C0 = torch.nn.Parameter(torch.Tensor([1.0]))

        self.skip_guidance_C_HR = Skip_guidance_HR(inplanes=filters[0])
        self.skip_guidance_C0 = Skip_guidance(inplanes=filters[0])
        self.skip_guidance_C1 = Skip_guidance(inplanes=filters[1])
        self.skip_guidance_C2 = Skip_guidance(inplanes=filters[2])
        self.skip_guidance_C3 = Skip_guidance(inplanes=filters[3])

        self.multi_scale_top = Dilat_Chnnel_Atten(filters[4])

        self.smooth_C4 = nn.Sequential(
            nn.Conv2d(filters[4], filters[4], kernel_size=3, padding=1, bias=False)
        )

        self.Conv_HR[0].weight = self.baseline.conv1.weight
        self.Conv_HR[1].weight = self.baseline.bn1.weight
        self.down_conv_HR[0].weight = self.baseline.layer1[0].conv1.weight
        self.down_conv_HR[1].weight = self.baseline.layer1[0].bn1.weight
        self.down_conv_C0[0].weight = self.baseline.layer1[0].conv1.weight
        self.down_conv_C0[1].weight = self.baseline.layer1[0].bn1.weight
        self.down_conv_C1[0].weight = self.baseline.layer2[0].conv1.weight
        self.down_conv_C1[1].weight = self.baseline.layer2[0].bn1.weight
        self.down_conv_C2[0].weight = self.baseline.layer3[0].conv1.weight
        self.down_conv_C2[1].weight = self.baseline.layer3[0].bn1.weight
        self.down_conv_C3[0].weight = self.baseline.layer4[0].conv1.weight
        self.down_conv_C3[1].weight = self.baseline.layer4[0].bn1.weight
        self.smooth_C4[0].weight = self.baseline.layer4[0].conv2.weight



        self.output_layer = nn.Sequential(
            nn.Conv2d(640, filters[0], kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(filters[0], num_classes, kernel_size=3, padding=1, bias=False),
        )




    def _upsample_add(self, x, y):

        '''Upsample and add two feature maps.
        '''

        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def upSample(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

    def forward(self, x):
        C0 = self.relu(self.bn1(self.conv1(x)))
        C1 = self.layer1(C0)
        C2 = self.layer2(C1)
        C3 = self.layer3(C2)
        C4 = self.layer4(C3)

        C_HR = self.Conv_HR(x)

        C_HR_down = self.down_apha_HR * self.down_conv_HR(C_HR)
        C0_down = self.down_apha_C0 * self.down_conv_C0(C0)
        C1_down = self.down_apha_C1 * self.down_conv_C1(C1)
        C2_down = self.down_apha_C2 * self.down_conv_C2(C2)
        C3_down = self.down_apha_C3 * self.down_conv_C3(C3)

        C4_up = self.up_apha_C4 * self.upSample_C4(C4)
        C3_up = self.up_apha_C3 * self.upSample_C3(C3)
        C2_up = self.up_apha_C2 * self.upSample_C2(C2)
        C1_up = self.up_apha_C1 * self.upSample_C1(C1)
        C0_up = self.up_apha_C0 * self.upSample_C0(C0)

        C_HR_branch_out = self.skip_guidance_C_HR(C_HR, C0_up)
        C0_branch_out = self.skip_guidance_C0(C0, C_HR_down, C1_up)
        C1_branch_out = self.skip_guidance_C1(C1, C0_down, C2_up)
        C2_branch_out = self.skip_guidance_C2(C2, C1_down, C3_up)
        C3_branch_out = self.skip_guidance_C3(C3, C2_down, C4_up)
        C4_branch_out = self.smooth_C4(C4 + C3_down)

        C4_p = self.conv1x1_4(self.multi_scale_top(C4_branch_out))

        C4_upSample = self.transpose_4(self._upsample_add(C4_p, C3_branch_out))

        C3_p = self.conv1x1_3(C4_upSample)
        C3_upSample = self.transpose_3(self._upsample_add(C3_p, C2_branch_out))

        C2_p = self.conv1x1_2(C3_upSample)
        C2_upSample = self.transpose_2(self._upsample_add(C2_p, C1_branch_out))

        C1_p = self.conv1x1_1(C2_upSample)
        C1_upSample = self.transpose_1(self._upsample_add(C1_p, C0_branch_out))

        C0_p = self.conv1x1_0(C1_upSample)
        C0_upSample = self.transpose_0(self._upsample_add(C0_p, C_HR_branch_out))

        C0_out = self.upSample(C0_p, x)
        C1_out = self.upSample(C1_p, x)
        C2_out = self.upSample(C2_p, x)
        C3_out = self.upSample(C3_p, x)
        C4_out = self.upSample(C4_p, x)

        result = self.output_layer(torch.cat([C0_upSample, C0_out, C1_out, C2_out, C3_out, C4_out], 1))
        result = F.sigmoid(result)
        return result


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("=========resnet18 OurModule_weight_init_test========")
    model = ResNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    print("=========resnet34 OurModule_weight_init_test========")
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


if __name__ == "__main__":
    import torch
    import numpy as np

    print('begin...')

    # model = Encoder_Path(64)
    model = resnet34(num_classes=3).cuda()
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    tmp = torch.randn(1, 3, 512, 512).cuda()
    y = torch.randn(1, 448, 448)

    import time

    start_time = time.time()
    print(model(tmp).shape)
    end_time = time.time()
    print("Time ==== {}".format(end_time - start_time))
    print('done')