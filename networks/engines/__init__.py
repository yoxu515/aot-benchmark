from networks.engines.aot_engine import AOTEngine, AOTInferEngine
from networks.engines.aotv2_engine import AOTv2Engine, AOTv2InferEngine
from networks.engines.aotv3_engine import AOTv3Engine, AOTv3InferEngine
from networks.engines.aost_engine import AOSTEngine, AOSTInferEngine
from networks.engines.paot_engine import PAOTEngine, PAOTInferEngine
from networks.engines.deaot_engine import DeAOTEngine, DeAOTInferEngine


def build_engine(name, phase='train', **kwargs):
    if name == 'aotengine':
        if phase == 'train':
            return AOTEngine(**kwargs)
        elif phase == 'eval':
            return AOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'aostengine':
        if phase == 'train':
            return AOSTEngine(**kwargs)
        elif phase == 'eval':
            return AOSTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'aotv2engine':
        if phase == 'train':
            return AOTv2Engine(**kwargs)
        elif phase == 'eval':
            return AOTv2InferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'aotv3engine':
        if phase == 'train':
            return AOTv3Engine(**kwargs)
        elif phase == 'eval':
            return AOTv3InferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'paotengine':
        if phase == 'train':
            return PAOTEngine(**kwargs)
        elif phase == 'eval':
            return PAOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'deaotengine':
        if phase == 'train':
            return DeAOTEngine(**kwargs)
        elif phase == 'eval':
            return DeAOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
