import torch.nn as nn

from networks.layers.transformer import LongShortTermTransformer, SharedLongShortTermTransformer
from networks.models.aot import AOT
from networks.decoders import build_decoder


class AOST(AOT):
    def __init__(self,
                 cfg,
                 encoder='mobilenetv2',
                 decoder='scalable_fpn',
                 lstt_type='vanilla'):
        super().__init__(cfg, encoder, 'fpn')
        self.lstt_layer_num = cfg.MODEL_LSTT_NUM

        if lstt_type == 'share':
            scalable_lstt = SharedLongShortTermTransformer
        elif lstt_type == 'vanilla':
            scalable_lstt = LongShortTermTransformer
        else:
            raise NotImplementedError

        self.LSTT = scalable_lstt(
            cfg.MODEL_LSTT_NUM,
            cfg.MODEL_ENCODER_EMBEDDING_DIM,
            cfg.MODEL_SELF_HEADS,
            cfg.MODEL_ATT_HEADS,
            emb_dropout=cfg.TRAIN_LSTT_EMB_DROPOUT,
            droppath=cfg.TRAIN_LSTT_DROPPATH,
            lt_dropout=cfg.TRAIN_LSTT_LT_DROPOUT,
            st_dropout=cfg.TRAIN_LSTT_ST_DROPOUT,
            droppath_lst=cfg.TRAIN_LSTT_DROPPATH_LST,
            droppath_scaling=cfg.TRAIN_LSTT_DROPPATH_SCALING,
            intermediate_norm=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            return_intermediate=True,
            block_version="v2")

        decoder_indims = []
        for layer_idx in range(self.lstt_layer_num):
            decoder_indim = cfg.MODEL_ENCODER_EMBEDDING_DIM * (
                layer_idx + 2
            ) if cfg.MODEL_DECODER_INTERMEDIATE_LSTT else cfg.MODEL_ENCODER_EMBEDDING_DIM
            decoder_indims.append(decoder_indim)

        self.decoder = build_decoder(
            decoder,
            in_dims=decoder_indims,
            out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
            decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
            shortcut_dims=cfg.MODEL_ENCODER_DIM,
            align_corners=cfg.MODEL_ALIGN_CORNERS)

        self.id_norm = nn.LayerNorm(cfg.MODEL_ENCODER_EMBEDDING_DIM)

        self._init_weight()

    def decode_id_logits(self, lstt_emb, shortcuts):
        n, c, h, w = shortcuts[-1].size()
        decoder_inputs = [shortcuts[-1]]
        for emb in lstt_emb:
            decoder_inputs.append(emb.view(h, w, n, c).permute(2, 3, 0, 1))
        pred_logit = self.decoder(decoder_inputs, shortcuts, len(lstt_emb) - 1)
        return pred_logit

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x)
        id_emb = self.id_norm(id_emb.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
        id_emb = self.id_dropout(id_emb)
        return id_emb
