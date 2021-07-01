from models.nets.UNet import UNet,UNet_Nested,UNet_Nested_dilated
# from models.nets.UNet_MAD import UNet_MAD
# from models.nets.UNET_MGPA import UNet_MGPA
# from models.nets.UNet_MAD_MGPA import UNet_MGPA_MAD
# from models.nets.Unet_spp_se import UNet_1
# from models.nets.Unet_spp_eca import UNet_2,UNet_2_2
# from models.nets.resnet_UNet import resnet34_UNet,resnet50_UNet
from models.nets.CPFNet import CPFNet
# from models.nets.UNet_modify import UNet_m
from models.nets.CE_Net import CE_Net
# from models.nets.test_module import CE_Net_ASPP
# from models.nets.AttU_Net import AttU_Net
# from models.nets.R2U_Net import R2U_Net
# from models.nets.pspnet import PSPNet
# from models.nets.DMsA import DSMA
# from models.nets.DMsA_cpf import DSMA_cpf
# from models.nets.MAD_M import MAD_M
# from models.nets.MAD_Mf_1 import MAD_Mf_1
# from models.nets.MAD_ONLY_MAD import MAD_ONLY_MAD
# from models.nets.MAD_ONLY_SPP import MAD_ONLY_SPP
# from models.nets.SPPResUNet import SPPResUNet
# from models.nets.OurModule_weight_init_test import resnet34 as ONet
# from models.nets.AdaptiveUNet_pp import AdaptiveUNet_PP
# from models.nets.ResUNet_pp import ResUNet_PP
# from models.nets.UNet_pp_row import UNet_PP
# from models.nets.Resunet import Resunet
# from models.nets.resunet_row import Resunet_row
# from models.nets.ISBI_Backbone import ISBI_Backbone
# from models.nets.Dynamic_SPP_UNet import Dynamic_SPP_UNet
# from models.nets.Dynamic_SPP_Caps_UNet import Dynamic_SPP_Caps_UNet
# from models.nets.Caps_ResUNet import Caps_ResUNet
# from models.nets.Caps_ResUNet_V1 import Caps_ResUNet_V1
# from models.nets.Trans_SegV0 import Trans_Seg_V0
# from models.nets.TMI_backbone import TMI_backbone
# from models.nets.modeling.deeplab import DeepLab



def net_builder(name,pretrained=False,n_class=1):
    # if name == 'resnet50_unet':
    #     net = resnet50_UNet(pretrained=pretrained)
    if name == 'unet':
        net = UNet(in_channels=1, n_classes=n_class, feature_scale=2)
    # elif name == 'unet_pp_raw':
    #     net = UNet_PP(in_channels=3, num_classes=n_class)
    # elif name == 'ISBI_Backbone':
    #     net = ISBI_Backbone(in_channels=3, num_classes=n_class)
    # elif name == 'resunet_pp*0.5-1':
    #     net = ResUNet_PP(num_classes=n_class)
    # elif name == 'resunet_pp-1':
    #     net = ResUNet_PP(num_classes=n_class)
    # elif name == 'resunet_pp_0':
    #     net = ResUNet_PP(num_classes=n_class)
    # elif name == 'Dynamic_SPP_UNet':
    #     net = Dynamic_SPP_UNet(in_channels=3,num_classes=n_class)
    # elif name == 'Dynamic_SPP_Caps_UNet':
    #     net = Dynamic_SPP_Caps_UNet(in_channels=3,num_classes=n_class)
    # elif name == 'Caps_ResUNet':
    #     net = Caps_ResUNet(in_channels=3,num_classes=n_class)
    # elif name == 'Caps_ResUNet_V1':
    #     net = Caps_ResUNet_V1(in_channels=3,num_classes=n_class)
    elif name == 'cpfnet':
        net = CPFNet(in_channels=3, out_planes=n_class)
    elif name == 'CEnet':
        net = CE_Net(in_channels=1, num_classes=n_class)
    # elif name == 'R2U_Net':
    #     net= R2U_Net(in_channels = 3,n_classes=n_class,feature_scale=2)
    # elif name == 'AttU_Net':
    #     net = AttU_Net(in_channels=3, num_classes=n_class, feature_scale=2)
    # elif name == "PSPNet":
    #     net = PSPNet(num_classes=n_class)
    # elif name == "Deeplab":
    #     net = DeepLab(num_classes=1)
    # elif name == "UNet_MGPA":
    #     net = UNet_MGPA(in_channels=3, n_classes=n_class, feature_scale=2)
    # elif name == "UNet_MAD":
    #     net = UNet_MAD(in_channels=3, n_classes=n_class, feature_scale=2)
    # elif name == "UNet_MGPA_MAD":
    #     net = UNet_MGPA_MAD(in_channels=3, n_classes=n_class, feature_scale=2)
    # elif name == "MAD_Mf2_1":
    #     net = MAD_Mf_1(num_classes=n_class)
    # elif name == "MAD_Mf2_2":
    #     net = MAD_Mf_1(num_classes=n_class)
    # elif name == "MAD_Mf2_3":
    #     net = MAD_Mf_1(num_classes=n_class)
    # elif name == "MAD_Mf2_4":
    #     net = MAD_Mf_1(num_classes=n_class)
    # elif name == "MAD_ONLY_MAD_1":
    #     net = MAD_ONLY_MAD(num_classes=n_class)
    # elif name == "MAD_ONLY_MAD_2":
    #     net = MAD_ONLY_MAD(num_classes=n_class)
    # elif name == "MAD_ONLY_MAD_3":
    #     net = MAD_ONLY_MAD(num_classes=n_class)
    # elif name == "MAD_ONLY_MAD_4":
    #     net = MAD_ONLY_MAD(num_classes=n_class)
    # elif name == "MAD_ONLY_SPP_1":
    #     net = MAD_ONLY_SPP(num_classes=n_class)
    # elif name == "MAD_ONLY_SPP_2":
    #     net = MAD_ONLY_SPP(num_classes=n_class)
    # elif name == "MAD_ONLY_SPP_3":
    #     net = MAD_ONLY_SPP(num_classes=n_class)
    # elif name == "MAD_ONLY_SPP_4":
    #     net = MAD_ONLY_SPP(num_classes=n_class)
    # elif name == 'resunet_one_skip':
    #     net = Resunet(in_channels=3, out_planes=n_class)
    # elif name == 'resunet_row':
    #     net = Resunet_row(in_channels=3, out_planes=n_class)
    # elif name == "SPPResUNet":
    #     net = SPPResUNet(num_classes=n_class)
    # elif name == "Trans_Seg_V0":
    #     net = Trans_Seg_V0(out_planes=1)
    # elif name == "TMI_backbone":
    #     net = TMI_backbone(out_planes=1)


    else:
        raise NameError("Unknow Model Name!")
    return net
