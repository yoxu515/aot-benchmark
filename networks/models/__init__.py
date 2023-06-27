from networks.models.aot import AOT
from networks.models.aotv2 import AOTv2
from networks.models.aotv3 import AOTv3
from networks.models.aost import AOST
from networks.models.deaot import DeAOT
from networks.models.paot import PAOT



def build_vos_model(name, cfg, **kwargs):

    if name == 'aot':
        return AOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == 'aotv2':
        return AOTv2(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == 'aotv3':
        return AOTv3(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == 'aost':  # LSTT layers share the parameters
        return AOST(cfg,
                    encoder=cfg.MODEL_ENCODER,
                    lstt_type='vanilla',
                    **kwargs)
    elif name == 'aost_share':  # LSTT layers do not share the parameters
        return AOST(cfg,
                    encoder=cfg.MODEL_ENCODER,
                    lstt_type='share',
                    **kwargs)
    elif name == 'deaot':
        return DeAOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == 'paot':
        return PAOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    else:
        raise NotImplementedError
