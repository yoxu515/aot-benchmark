from .default import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'SwinB_PAOT'
        self.MODEL_VOS = 'paot'
        self.MODEL_ENGINE = 'paotengine'
        self.MODEL_ENCODER = 'swin_base'
        self.MODEL_ALIGN_CORNERS = False
        
        self.MODEL_ENCODER_DIM = [128, 256, 512, 512]  # 4x, 8x, 16x, 16x
        
        self.MODEL_DECODER_INTERMEDIATE_LSTT = True

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5
        self.TRAIN_PANO = True
        self.TEST_PANO = True
        self.MODEL_MAX_THING_NUM = 10
        self.MODEL_MAX_STUFF_NUM = 6
        self.MODEL_SEP_ID_BANK = True
        self.MODEL_USE_ID_BANK_POST_CONV = True
