from networks.encoders.mobilenetv2 import MobileNetV2
from networks.encoders.mobilenetv3 import MobileNetV3Large
from networks.encoders.resnet import ResNet101, ResNet50, ResNet50_CLIP, ResNet18
from networks.encoders.resnest import resnest
from networks.encoders.swin import build_swin_model
from networks.layers.normalization import FrozenBatchNorm2d,LayerNorm2d
from torch import nn


def build_encoder(name, frozen_bn=True, freeze_at=-1, in_channel=3, use_ln=False):
    if frozen_bn:
        BatchNorm = FrozenBatchNorm2d
    elif use_ln:
        BatchNorm = LayerNorm2d
    else:
        BatchNorm = nn.BatchNorm2d

    if name == 'mobilenetv2':
        return MobileNetV2(16, BatchNorm, freeze_at=freeze_at)
    elif name == 'mobilenetv3':
        return MobileNetV3Large(16, BatchNorm, freeze_at=freeze_at)
    elif name == 'resnet18':
        return ResNet18(16, BatchNorm, freeze_at=freeze_at, in_channel=in_channel)
    elif name == 'resnet50':
        return ResNet50(16, BatchNorm, freeze_at=freeze_at)
    elif name == 'resnet101':
        return ResNet101(16, BatchNorm, freeze_at=freeze_at)
    elif name == 'resnest50':
        return resnest.resnest50(norm_layer=BatchNorm,
                                 dilation=2,
                                 freeze_at=freeze_at)
    elif name == 'resnest101':
        return resnest.resnest101(norm_layer=BatchNorm,
                                  dilation=2,
                                  freeze_at=freeze_at)
    elif 'swin' in name:
        return build_swin_model(name, freeze_at=freeze_at, in_chans=in_channel)
    elif name == 'resnet50_clip':
        return ResNet50_CLIP(16, BatchNorm, freeze_at=freeze_at)
    else:
        raise NotImplementedError
