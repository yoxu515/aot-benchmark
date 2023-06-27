from .default import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'AOST_Share'
        self.MODEL_VOS = 'aost_share'
        self.MODEL_ENGINE = 'aostengine'

        self.MODEL_ENCODER = 'mobilenetv2'

        self.MODEL_LSTT_NUM = 3
        self.MODEL_DECODER_INTERMEDIATE_LSTT = False

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5
