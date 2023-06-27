import numpy as np

from utils.image import one_hot_mask

from networks.layers.basic import seq_to_2d
from networks.engines.aot_engine import AOTEngine, AOTInferEngine


class AOSTEngine(AOTEngine):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 layer_loss_scaling_ratio=2.):
        super().__init__(aot_model, gpu_id, long_term_mem_gap,
                         short_term_mem_skip)
        self.layer_loss_scaling_ratio = layer_loss_scaling_ratio

    def update_short_term_memory(self, curr_mask, curr_id_emb=None):
        if curr_id_emb is None:
            if len(curr_mask.size()) == 3 or curr_mask.size()[0] == 1:
                curr_one_hot_mask = one_hot_mask(curr_mask, self.max_obj_num)
            else:
                curr_one_hot_mask = curr_mask
            curr_id_emb = self.assign_identity(curr_one_hot_mask)

        lstt_curr_memories = self.curr_lstt_output[1]
        lstt_curr_memories_2d = []
        for layer_idx in range(len(lstt_curr_memories)):
            curr_k, curr_v = lstt_curr_memories[layer_idx][
                0], lstt_curr_memories[layer_idx][1]
            if 'share' in self.cfg.MODEL_VOS:
                lstt_layer = self.AOT.LSTT.layer
            else:
                lstt_layer = self.AOT.LSTT.layers[layer_idx]
            curr_k, curr_v = lstt_layer.fuse_key_value_id(
                curr_k, curr_v, curr_id_emb)
            lstt_curr_memories[layer_idx][0], lstt_curr_memories[layer_idx][
                1] = curr_k, curr_v
            lstt_curr_memories_2d.append([
                seq_to_2d(lstt_curr_memories[layer_idx][0], self.enc_size_2d),
                seq_to_2d(lstt_curr_memories[layer_idx][1], self.enc_size_2d)
            ])

        self.short_term_memories_list.append(lstt_curr_memories_2d)
        self.short_term_memories_list = self.short_term_memories_list[
            -self.short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]

        if self.frame_step - self.last_mem_step >= self.long_term_mem_gap:
            self.update_long_term_memory(lstt_curr_memories)
            self.last_mem_step = self.frame_step

    def generate_loss_mask(self, gt_mask, step, return_prob=False):
        self.curr_lstt_output = list(self.curr_lstt_output)
        curr_lstt_embs = self.curr_lstt_output[0]
        layer_num = len(curr_lstt_embs)
        all_layer_loss = 0
        all_layer_weight = 0

        for layer_idx in range(layer_num):
            temp_curr_lstt_embs = curr_lstt_embs[:(layer_idx + 1)]
            self.curr_lstt_output[0] = temp_curr_lstt_embs
            self.decode_current_logits()
            layer_loss_weight = self.layer_loss_scaling_ratio**layer_idx
            layer_loss = self.calculate_current_loss(gt_mask, step)
            all_layer_loss = all_layer_loss + layer_loss * layer_loss_weight
            all_layer_weight += layer_loss_weight

        loss = all_layer_loss / all_layer_weight

        if return_prob:
            mask, prob = self.predict_current_mask(return_prob=True)
            return loss, mask, prob
        else:
            mask = self.predict_current_mask()
            return loss, mask


class AOSTInferEngine(AOTInferEngine):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 max_aot_obj_num=None):
        super().__init__(aot_model, gpu_id, long_term_mem_gap,
                         short_term_mem_skip, max_aot_obj_num)

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        aot_num = max(np.ceil(obj_nums / self.max_aot_obj_num), 1)
        while (aot_num > len(self.aot_engines)):
            new_engine = AOSTEngine(self.AOT, self.gpu_id,
                                    self.long_term_mem_gap,
                                    self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)

        separated_masks = self.separate_mask(mask)
        img_embs = None
        for aot_engine, separated_mask in zip(self.aot_engines,
                                              separated_masks):
            aot_engine.add_reference_frame(img,
                                           separated_mask,
                                           obj_nums=[self.max_aot_obj_num],
                                           frame_step=frame_step,
                                           img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

        self.update_size()
