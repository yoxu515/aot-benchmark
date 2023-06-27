from .default import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'AOTv2'
        self.MODEL_VOS = 'aotv2'
        self.MODEL_ENGINE = 'aotv2engine'
        self.MODEL_ENCODER = 'mobilenetv2'
        self.MODEL_DECODER_INTERMEDIATE_LSTT = False
        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5
