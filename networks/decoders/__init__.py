from networks.decoders.fpn import FPNSegmentationHead, ScalableFPNSegmentationHead


def build_decoder(name, **kwargs):

    if name == 'fpn':
        return FPNSegmentationHead(**kwargs)
    elif name == 'scalable_fpn':
        return ScalableFPNSegmentationHead(**kwargs)
    else:
        raise NotImplementedError
