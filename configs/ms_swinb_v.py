import os
from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='SwinB_AOTv3',stage='default'):
        super().__init__(exp_name, model)
        if stage == 'default':
            self.STAGE_NAME = 'PRE_YTB_DAV'
        else:
            self.STAGE_NAME = stage
        if self.STAGE_NAME == 'PRE':
            self.DATASETS = ['static']

            self.DATA_DYNAMIC_MERGE_PROB = 1.0

            self.TRAIN_LR = 4e-4
            self.TRAIN_LR_MIN = 2e-5
            self.TRAIN_WEIGHT_DECAY = 0.03
            self.TRAIN_SEQ_TRAINING_START_RATIO = 1.0
            self.TRAIN_AUX_LOSS_RATIO = 0.1

            self.init_dir(data='./datasets',root='./results',eval='./')
            self.MODEL_ENCODER_PRETRAIN = './pretrain_models/cascade_mask_rcnn_swin_base_patch4_window7_backbone.pth'
            self.PRETRAIN_MODEL = './pretrain_models/cascade_mask_rcnn_swin_base_patch4_window7_backbone.pth'

            self.TRAIN_MS_LSTT_DROPPATH = [0.1,0.1,0.5,0.1]
            self.TRAIN_MS_LSTT_DROPPATH_LST = [False,False,True,False]
            self.TRAIN_MS_LSTT_LT_DROPOUT = [0.,0.,0.,0.]
            self.TRAIN_MS_LSTT_ST_DROPOUT = [0.,0.,0.,0.]

            self.DATA_PRE_STRONG_AUG = True
            self.DATA_TPS_PROB = 0.3
            self.DATA_TPS_SCALE = 0.02


        elif self.STAGE_NAME == 'PRE_YTB_DAV':

            self.DATASETS = ['vipseg']
            self.init_dir(data='./datasets',root='./results',eval='./')

            self.DATA_DYNAMIC_MERGE_PROB_VIP = 0.0
            self.DATA_RANDOM_GAP_VIP = 3
            self.DATA_YTB_REPEAT = 0
            
            pretrain_exp = self.EXP_NAME
            pretrain_stage = 'PRE'
            pretrain_ckpt = 'save_step_100000.pth'
            self.PRETRAIN_FULL = True  # if False, load encoder only
            self.PRETRAIN_MODEL = os.path.join(self.DIR_ROOT, 'result',
                                            pretrain_exp, pretrain_stage,
                                            'ema_ckpt', pretrain_ckpt)
            
            self.TRAIN_MS_LSTT_DROPPATH = [0.1,0.1,0.2,0.1]
            self.TRAIN_MS_LSTT_DROPPATH_LST = [False,False,True,False]
            self.TRAIN_MS_LSTT_LT_DROPOUT = [0.,0.,0.,0.]
            self.TRAIN_MS_LSTT_ST_DROPOUT = [0.,0.,0.,0.]

            self.TRAIN_SAVE_MED_STEP = 10000
            self.TRAIN_START_SAVE_MED_RATIO = 0.3

            self.DATA_RANDOM_GAUSSIAN_BLUR = 0.3
            self.DATA_RANDOM_GRAYSCALE = 0.2
            self.DATA_RANDOM_COLOR_JITTER = 0.8

            self.TRAIN_AUX_LOSS_WEIGHT = 1.0
            self.TRAIN_AUX_LOSS_RATIO = 0.75

        # multi-scale param
        self.MODEL_MS_SCALES = [16,16,8,4]

        self.MODEL_MS_LSTT_NUMS = [2,1,1,0]
        self.MODEL_MS_ENCODER_EMBEDDING_DIMS = [256,256,256,128]
        self.MODEL_MS_SELF_HEADS = [8,8,4,1]
        self.MODEL_MS_ATT_HEADS = [1,1,1,1]
        
        self.MODEL_MS_FEEDFOWARD_DIMS = [1024,1024,1024,512]
        self.MODEL_MS_GLOBAL_DILATIONS = [1,1,2,2]

        self.MODEL_DECODER_RES = True
        self.MODEL_DECODER_RES_IN = True
        
        self.TRAIN_MS_LSTT_MEMORY_DILATION = True # save memory

