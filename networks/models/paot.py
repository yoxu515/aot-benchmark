import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.encoders import build_encoder
from networks.decoders import build_decoder
from networks.layers.position import PositionEmbeddingSine
from networks.layers.transformer import MSLongShortTermTransformer
from networks.layers.basic import ConvGN
from networks.encoders.resnet import BasicBlock
from networks.layers.normalization import LayerNorm2d


class PAOT(nn.Module):
    def __init__(self, cfg, encoder='mobilenetv2', decoder='fpn'):
        super().__init__()

        self.cfg = cfg
        self.max_stuff_num = cfg.MODEL_MAX_STUFF_NUM
        self.max_thing_num = cfg.MODEL_MAX_THING_NUM
        self.epsilon = cfg.MODEL_EPSILON

        self.encoder = build_encoder(encoder,
                                     frozen_bn=cfg.MODEL_FREEZE_BN,
                                     freeze_at=cfg.TRAIN_ENCODER_FREEZE_AT)
        
        self.adapters = nn.ModuleList()
        for s in range(len(cfg.MODEL_ENCODER_DIM)):
            self.adapters.append(nn.Conv2d(cfg.MODEL_ENCODER_DIM[-(s+1)], cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[s], 1))

        self.pos_generators = nn.ModuleList()
        for s in range(len(cfg.MODEL_ENCODER_DIM)):
            self.pos_generators.append(PositionEmbeddingSine(
                cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[s]//2, normalize=True))

        self.MSLSTT = MSLongShortTermTransformer(
            cfg.MODEL_MS_LSTT_NUMS,
            cfg.MODEL_ENCODER_DIM,
            cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS,
            cfg.MODEL_MS_SELF_HEADS,
            cfg.MODEL_MS_ATT_HEADS,
            dims_feedforward=cfg.MODEL_MS_FEEDFOWARD_DIMS,
            global_dilations=cfg.MODEL_MS_GLOBAL_DILATIONS,
            local_dilations=cfg.MODEL_MS_LOCAL_DILATIONS,
            memory_dilation=cfg.TRAIN_MS_LSTT_MEMORY_DILATION,
            conv_dilation=cfg.MODEL_MS_CONV_DILATION,
            att_dims=cfg.MODEL_MS_ATT_DIMS,
            emb_dropouts=cfg.TRAIN_MS_LSTT_EMB_DROPOUTS,
            droppath=cfg.TRAIN_MS_LSTT_DROPPATH,
            lt_dropout=cfg.TRAIN_MS_LSTT_LT_DROPOUT,
            st_dropout=cfg.TRAIN_MS_LSTT_ST_DROPOUT,
            droppath_lst=cfg.TRAIN_MS_LSTT_DROPPATH_LST,
            droppath_scaling=cfg.TRAIN_MS_LSTT_DROPPATH_SCALING,
            intermediate_norm=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            return_intermediate=True,
            align_corners=cfg.MODEL_ALIGN_CORNERS,
            decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            decoder_res=cfg.MODEL_DECODER_RES,
            decoder_res_in=cfg.MODEL_DECODER_RES_IN,
            use_relative_v=cfg.MODEL_USE_RELATIVE_V,
            use_self_pos=cfg.MODEL_USE_SELF_POS,
            topk=cfg.TEST_TOP_K)
        
        if cfg.MODEL_USE_ID_ENCODER:
            self.id_encoder = build_encoder(cfg.MODEL_ID_ENCODER,
                                        frozen_bn=cfg.MODEL_ID_ENCODER_FROZEN_BN,
                                        freeze_at=cfg.MODEL_ID_ENCODER_FREEZE_AT,
                                        in_channel=cfg.MODEL_MAX_OBJ_NUM+1,
                                        use_ln=cfg.MODEL_ID_ENCODER_USE_LN)
            self.id_encoder_adaptors  = nn.ModuleList()
            for i,d in enumerate(cfg.MODEL_ID_ENCODER_DIM[::-1]):
                self.id_encoder_adaptors.append(
                    nn.Conv2d(
                        d,cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i],kernel_size=1,
                    )
                )
        if cfg.MODEL_SEP_ID_BANK:
            self.thing_id_banks = nn.ModuleList()
            self.stuff_id_banks = nn.ModuleList()
        else:
            self.patch_wise_id_banks = nn.ModuleList()
        self.id_norms = nn.ModuleList()
        if cfg.MODEL_USE_ID_BANK_POST_CONV:
            self.id_post_convs = nn.ModuleList()
        scales = cfg.MODEL_MS_SCALES
        for i,s in enumerate(scales):
            if cfg.MODEL_ALIGN_CORNERS:
                if cfg.MODEL_SEP_ID_BANK:
                    self.thing_id_banks.append(nn.Conv2d(
                        cfg.MODEL_MAX_THING_NUM,
                        cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i],
                        kernel_size=s+1,
                        stride=s,
                        padding=s//2))
                    self.stuff_id_banks.append(nn.Conv2d(
                        cfg.MODEL_MAX_STUFF_NUM,
                        cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i],
                        kernel_size=s+1,
                        stride=s,
                        padding=s//2))
                else:
                    self.patch_wise_id_banks.append(nn.Conv2d(
                        cfg.MODEL_MAX_STUFF_NUM + cfg.MODEL_MAX_THING_NUM,
                        cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i],
                        kernel_size=s+1,
                        stride=s,
                        padding=s//2))
            else:
                if cfg.MODEL_SEP_ID_BANK:
                    self.thing_id_banks.append(nn.Conv2d(
                        cfg.MODEL_MAX_THING_NUM,
                        cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i],
                        kernel_size=s,
                        stride=s,
                        padding=0))
                    self.stuff_id_banks.append(nn.Conv2d(
                        cfg.MODEL_MAX_STUFF_NUM,
                        cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i],
                        kernel_size=s,
                        stride=s,
                        padding=0))
                else:
                    self.patch_wise_id_banks.append(nn.Conv2d(
                        cfg.MODEL_MAX_STUFF_NUM + cfg.MODEL_MAX_THING_NUM,
                        cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i],
                        kernel_size=s,
                        stride=s,
                        padding=0))
            
            if cfg.MODEL_USE_ID_BANK_POST_CONV:
                self.id_norms.append(nn.LayerNorm(cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i] * 2))
                self.id_post_convs.append(BasicBlock(cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i]*2,
                                            cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i],BatchNorm=LayerNorm2d))
            else:
                self.id_norms.append(nn.LayerNorm(cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i]))
        self.id_dropout = nn.Dropout(cfg.TRAIN_LSTT_ID_DROPOUT, True)
        
        if cfg.MODEL_DECODER_INTERMEDIATE_LSTT:
            self.conv_output1 = ConvGN(cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[-1] * (cfg.MODEL_MS_LSTT_NUMS[-1] + 1),
                                    cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[-1], 3)
        else:
            self.conv_output1 = ConvGN(cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[-1],
                                    cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[-1], 3)
        self.conv_output2 = nn.Conv2d(cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[-1], cfg.MODEL_MAX_THING_NUM+cfg.MODEL_MAX_STUFF_NUM, 1)
        
        if cfg.TRAIN_INTERMEDIATE_PRED_LOSS:
            decoder_indim = cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[0] * \
                (cfg.MODEL_MS_LSTT_NUMS[0] + 1) \
                if cfg.MODEL_DECODER_INTERMEDIATE_LSTT else cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[0]
            self.decoder = build_decoder(
                decoder,
                in_dim=decoder_indim,
                out_dim=cfg.MODEL_MAX_THING_NUM+cfg.MODEL_MAX_STUFF_NUM,
                decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
                hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
                shortcut_dims=cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS,
                align_corners=cfg.MODEL_ALIGN_CORNERS,
                use_adapters=False)
        
        
        self._init_weight()

    def get_pos_embs(self, xs):
        pos_embs = []
        for i,generator in enumerate(self.pos_generators):
            pos_emb = generator(xs[-(i+1)])
            pos_embs.append(pos_emb)
        return pos_embs

    def get_id_embs(self,one_hot_stuff, one_hot_thing):
        '''
        input: one_hot_label for thing and stuff
        generate id embedding for thing and stuff
        '''
        id_embs = []
        for i in range(len(self.cfg.MODEL_MS_SCALES)):
            if not self.cfg.MODEL_SEP_ID_BANK:
                one_hot = torch.cat([one_hot_stuff,one_hot_thing],dim=1)
                id_emb = self.patch_wise_id_banks[i](one_hot)
                id_emb = self.id_norms[i](id_emb.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
            else:
                thing_id_emb = self.thing_id_banks[i](one_hot_thing)
                stuff_id_emb = self.stuff_id_banks[i](one_hot_stuff)
                if self.cfg.MODEL_USE_ID_BANK_POST_CONV:
                    id_emb = torch.cat([stuff_id_emb,thing_id_emb],dim=1)
                    id_emb = self.id_norms[i](id_emb.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
                    id_emb = self.id_post_convs[i](id_emb)
                else:
                    id_emb = thing_id_emb + stuff_id_emb
                    id_emb = self.id_norms[i](id_emb.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
            id_emb = self.id_dropout(id_emb)
            id_embs.append(id_emb)
        return id_embs

    def encode_image(self, img):
        xs = self.encoder(img)
        for i,adapter in enumerate(self.adapters):
            xs[-(i+1)] = adapter(xs[-(i+1)])
        return xs
    
    def decode_id_logits(self, lstt_embs):
        output = F.relu(self.conv_output1(lstt_embs[-1]))
        output = self.conv_output2(output)
        return output
    def decode_med_logits(self,lstt_med_embs,encoder_embs):
        n, c, h, w = encoder_embs[-1].size()
        decoder_inputs = []
        for emb in lstt_med_embs:
            decoder_inputs.append(emb.view(h, w, n, c).permute(2, 3, 0, 1))
        med_pred = self.decoder(decoder_inputs, encoder_embs)
        return med_pred
    
    def LSTT_forward(self,
                     curr_embs,
                     long_term_memories,
                     short_term_memories,
                     curr_id_embs=None,
                     pos_embs=None,
                     sizes_2d: list=[(30,30),(30,30),(59,59),(117,117)]):
            
        lstt_embs, lstt_memories = self.MSLSTT(curr_embs, long_term_memories,
                                             short_term_memories, curr_id_embs,
                                             pos_embs, sizes_2d)
        lstt_curr_memories, lstt_long_memories, lstt_short_memories = zip(
            *lstt_memories)
        return lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories
   

    def _init_weight(self):
        for adapter in self.adapters:
            nn.init.xavier_uniform_(adapter.weight)
        for s in range(len(self.cfg.MODEL_MS_SCALES)):
            if not self.cfg.MODEL_SEP_ID_BANK:
                nn.init.orthogonal_(
                    self.patch_wise_id_banks[s].weight.view(
                        self.cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[s], -1).permute(0, 1),
                    gain=17**-2 if self.cfg.MODEL_ALIGN_CORNERS else 16**-2)
            else:
                nn.init.orthogonal_(
                    self.thing_id_banks[s].weight.view(
                        self.cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[s], -1).permute(0, 1),
                    gain=17**-2 if self.cfg.MODEL_ALIGN_CORNERS else 16**-2)
                nn.init.orthogonal_(
                    self.stuff_id_banks[s].weight.view(
                        self.cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[s], -1).permute(0, 1),
                    gain=17**-2 if self.cfg.MODEL_ALIGN_CORNERS else 16**-2)
        nn.init.xavier_uniform_(self.conv_output2.weight)
