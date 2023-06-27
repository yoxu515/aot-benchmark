# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
# from .swin_transformer_v2 import SwinTransformerV2


def build_swin_model(model_type, freeze_at=0, in_chans=3):
    if model_type == 'swin_base':
        model = SwinTransformer(embed_dim=128,
                                depths=[2, 2, 18, 2],
                                num_heads=[4, 8, 16, 32],
                                window_size=7,
                                drop_path_rate=0.3,
                                out_indices=(0, 1, 2),
                                ape=False,
                                patch_norm=True,
                                frozen_stages=freeze_at,
                                use_checkpoint=False)
    elif model_type == 'swinw_base':
        model = SwinTransformer(embed_dim=128,
                                pretrain_img_size=384,
                                depths=[2, 2, 18, 2],
                                num_heads=[4, 8, 16, 32],
                                window_size=12,
                                drop_path_rate=0.2,
                                out_indices=(0, 1, 2),
                                ape=False,
                                patch_norm=True,
                                frozen_stages=freeze_at,
                                use_checkpoint=False)
    elif model_type == 'swin_large':
        model = SwinTransformer(embed_dim=192,
                                depths=[2, 2, 18, 2],
                                num_heads=[6, 12, 24, 48],
                                window_size=7,
                                drop_path_rate=0.3,
                                out_indices=(0, 1, 2),
                                ape=False,
                                patch_norm=True,
                                frozen_stages=freeze_at,
                                use_checkpoint=False)
    elif model_type == 'swin_small':
        model = SwinTransformer(embed_dim=96,
                                depths=[2, 2, 18, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                drop_path_rate=0.3,
                                out_indices=(0, 1, 2),
                                ape=False,
                                patch_norm=True,
                                frozen_stages=freeze_at,
                                use_checkpoint=False)
    elif model_type == 'swin_tiny':
        model = SwinTransformer(in_chans=in_chans,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                drop_path_rate=0.2,
                                out_indices=(0, 1, 2),
                                ape=False,
                                patch_norm=True,
                                frozen_stages=freeze_at,
                                use_checkpoint=False)
    # elif model_type == 'swinv2_base':
    #     model = SwinTransformerV2(embed_dim=128,
    #                                 img_size=192,
    #                                 depths=[ 2, 2, 18, 2 ],
    #                                 num_heads=[ 4, 8, 16, 32 ],
    #                                 window_size=7,
    #                                 drop_path_rate=0.2,
    #                                 ape=False,
    #                                 patch_norm=True,
    #                                 frozen_stages=freeze_at,
    #                                 use_checkpoint=False)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
