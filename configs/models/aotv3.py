from .default import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'AOTv3'
        self.MODEL_VOS = 'aotv3'
        self.MODEL_ENGINE = 'aotv3engine'
        self.MODEL_ENCODER = 'mobilenetv2'
        
        self.MODEL_DECODER_INTERMEDIATE_LSTT = True
