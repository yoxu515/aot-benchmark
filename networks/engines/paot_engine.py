import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.math import generate_permute_matrix,generate_rearrange_matrix
from utils.image import one_hot_mask, split_stuff_thing_mask

from networks.layers.basic import seq_to_2d
from networks.models.paot import PAOT


class PAOTEngine(nn.Module):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1):
        super().__init__()

        self.cfg = aot_model.cfg
        self.align_corners = aot_model.cfg.MODEL_ALIGN_CORNERS
        self.AOT = aot_model

        self.max_stuff_num = aot_model.max_stuff_num
        self.max_thing_num = aot_model.max_thing_num
        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.losses = None

        self.restart_engine()

    def forward(self,
                all_frames,
                all_masks,
                batch_size,
                obj_nums, # [b,2] stuff num, thing num
                step=0,
                tf_board=False,
                use_prev_pred=False,
                enable_prev_frame=False,
                use_prev_prob=False):  # only used for training
        if self.losses is None:
            self._init_losses()
        obj_nums = obj_nums[0]
        self.obj_mapping = obj_nums[1] # [B dict]
        
        self.freeze_id = True if use_prev_pred else False
        aux_weight = self.aux_weight * max(self.aux_step - step,
                                           0.) / self.aux_step

        self.offline_encoder(all_frames, all_masks)

        self.add_reference_frame(frame_step=0, obj_nums=obj_nums)

        grad_state = torch.no_grad if aux_weight == 0 else torch.enable_grad
        with grad_state():
            ref_aux_loss, ref_aux_mask = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step)

        aux_losses = [ref_aux_loss]
        aux_masks = [ref_aux_mask]

        curr_losses, curr_masks = [], []
        if enable_prev_frame:
            self.set_prev_frame(frame_step=1)
            with grad_state():
                prev_aux_loss, prev_aux_mask = self.generate_loss_mask(
                    self.offline_masks[self.frame_step], step)
            aux_losses.append(prev_aux_loss)
            aux_masks.append(prev_aux_mask)
        else:
            self.match_propogate_one_frame()
            curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step, return_prob=True)
            self.update_short_term_memory(
                curr_mask if not use_prev_prob else curr_prob,
                None if use_prev_pred else self.assign_identity(
                    self.offline_stuff_one_hot[self.frame_step],self.offline_thing_one_hot[self.frame_step]))
            curr_losses.append(curr_loss)
            curr_masks.append(curr_mask)

        self.match_propogate_one_frame()
        curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
            self.offline_masks[self.frame_step], step, return_prob=True)
        curr_losses.append(curr_loss)
        curr_masks.append(curr_mask)
        for _ in range(self.total_offline_frame_num - 3):
            self.update_short_term_memory(
                curr_mask if not use_prev_prob else curr_prob,
                None if use_prev_pred else self.assign_identity(
                    self.offline_stuff_one_hot[self.frame_step],self.offline_thing_one_hot[self.frame_step]))
            self.match_propogate_one_frame()
            curr_loss, curr_mask, curr_prob = self.generate_loss_mask(
                self.offline_masks[self.frame_step], step, return_prob=True)
            curr_losses.append(curr_loss)
            curr_masks.append(curr_mask)

        aux_loss = torch.cat(aux_losses, dim=0).mean(dim=0)
        pred_loss = torch.cat(curr_losses, dim=0).mean(dim=0)

        loss = aux_weight * aux_loss + pred_loss

        all_pred_mask = aux_masks + curr_masks

        all_frame_loss = aux_losses + curr_losses

        boards = {'image': {}, 'scalar': {}}

        return loss, all_pred_mask, all_frame_loss, boards

    def _init_losses(self):
        cfg = self.cfg

        from networks.layers.loss import CrossEntropyLoss, SoftJaccordLoss
        bce_loss = CrossEntropyLoss(
            cfg.TRAIN_TOP_K_PERCENT_PIXELS,
            cfg.TRAIN_HARD_MINING_RATIO * cfg.TRAIN_TOTAL_STEPS)
        iou_loss = SoftJaccordLoss()

        losses = [bce_loss, iou_loss]
        loss_weights = [0.5, 0.5]

        self.losses = nn.ModuleList(losses)
        self.loss_weights = loss_weights
        self.aux_weight = cfg.TRAIN_AUX_LOSS_WEIGHT
        self.aux_step = cfg.TRAIN_TOTAL_STEPS * cfg.TRAIN_AUX_LOSS_RATIO + 1e-5

    def encode_one_img_mask(self, img=None, mask=None, frame_step=-1):
        '''
        mask: (B,3,H,W) for obj,stuff,thing
        '''
        if frame_step == -1:
            frame_step = self.frame_step

        if self.enable_offline_enc:
            curr_enc_embs = self.offline_enc_embs[frame_step]
        elif img is None:
            curr_enc_embs = None
        else:
            curr_enc_embs = self.AOT.encode_image(img)

        if mask is not None:
            stuff_mask,thing_mask = split_stuff_thing_mask(mask, self.obj_mapping)
            curr_stuff_one_hot = one_hot_mask(stuff_mask,self.max_stuff_num, add_bk=False)
            curr_thing_one_hot = one_hot_mask(thing_mask,self.max_thing_num, add_bk=False)
        elif self.enable_offline_enc:
            curr_stuff_one_hot = self.offline_stuff_one_hot[frame_step]
            curr_thing_one_hot = self.offline_thing_one_hot[frame_step]
        else:
            curr_stuff_one_hot = None
            curr_thing_one_hot = None

        return curr_enc_embs, curr_stuff_one_hot, curr_thing_one_hot

    def offline_encoder(self, all_frames, all_masks=None):
        self.enable_offline_enc = True
        self.offline_frames = all_frames.size(0) // self.batch_size

        # extract backbone features
        self.offline_enc_embs = self.split_frames(
            self.AOT.encode_image(all_frames), self.batch_size)
        self.total_offline_frame_num = len(self.offline_enc_embs)

        if all_masks is not None:
            # extract mask embeddings
            self.offline_one_hot_masks = one_hot_mask(all_masks, self.max_stuff_num + self.max_thing_num, add_bk=False) #(N,Ct,H,W)
            self.offline_masks = list(
                torch.split(all_masks, self.batch_size, dim=0)) #(frame_step,B)
            self.offline_stuff_one_hot = []
            self.offline_thing_one_hot = []
            for mask in self.offline_masks: # frame_step
                stuff_mask,thing_mask = split_stuff_thing_mask(mask, self.obj_mapping)
                self.offline_stuff_one_hot.append(one_hot_mask(stuff_mask,self.max_stuff_num, add_bk=False))
                self.offline_thing_one_hot.append(one_hot_mask(thing_mask,self.max_thing_num, add_bk=False))

        if self.input_size_2d is None:
            self.update_sizes(all_frames.size()[2:],
                             [emb.size()[2:] for emb in self.offline_enc_embs[0]])

    def assign_identity(self, one_hot_stuff, one_hot_thing):
        if self.enable_id_shuffle:
            one_hot_stuff = torch.einsum('bohw,bot->bthw', one_hot_stuff,
                                        self.id_shuffle_matrix_stuff)
            one_hot_thing = torch.einsum('bohw,bot->bthw', one_hot_thing,
                                        self.id_shuffle_matrix_thing)

        id_embs = self.AOT.get_id_embs(one_hot_stuff,one_hot_thing)
        for i in range(len(id_embs)):
            id_embs[i] = id_embs[i].view(
            self.batch_size, -1, self.enc_hws[-(i+1)]).permute(2, 0, 1)

            if self.training and self.freeze_id:
                id_embs[i] = id_embs[i].detach()

        return id_embs

    def split_frames(self, xs, chunk_size):
        new_xs = []
        for x in xs:
            all_x = list(torch.split(x, chunk_size, dim=0))
            new_xs.append(all_x)
        return list(zip(*new_xs))

    def add_reference_frame(self,
                            img=None,
                            mask=None,
                            frame_step=-1,
                            obj_nums=None,
                            img_embs=None):
        self.generate_split_matrix()
        if self.obj_nums is None and obj_nums is None:
            print('No objects for reference frame!')
            exit()
        elif obj_nums is not None:
            self.obj_nums = obj_nums

        if frame_step == -1:
            frame_step = self.frame_step

        if img_embs is None:
            curr_enc_embs, curr_stuff_one_hot, curr_thing_one_hot = self.encode_one_img_mask(
                img, mask, frame_step)
        else:
            _, curr_stuff_one_hot, curr_thing_one_hot = self.encode_one_img_mask(
                None, mask, frame_step)
            curr_enc_embs = img_embs

        if curr_enc_embs is None:
            print('No image for reference frame!')
            exit()

        if curr_thing_one_hot is None or curr_stuff_one_hot is None:
            print('No mask for reference frame!')
            exit()

        if self.input_size_2d is None:
            self.update_sizes(img.size()[2:], [emb.size()[2:] for emb in curr_enc_embs])

        self.curr_enc_embs = curr_enc_embs
        self.curr_stuff_one_hot = curr_stuff_one_hot
        self.curr_thing_one_hot = curr_thing_one_hot

        if self.pos_embs is None:
            self.pos_embs = self.AOT.get_pos_embs(curr_enc_embs)
            for i in range(len(curr_enc_embs)):
                self.pos_embs[i] = self.pos_embs[i].expand(
                    self.batch_size, -1, -1,
                    -1).view(self.batch_size, -1, self.enc_hws[-(i+1)]).permute(2, 0, 1)

        
        curr_id_embs = self.assign_identity(curr_stuff_one_hot, curr_thing_one_hot)
        self.curr_id_embs = curr_id_embs
        
        # self matching and propagation
        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      None,
                                                      None,
                                                      curr_id_embs,
                                                      pos_embs=self.pos_embs,
                                                      sizes_2d=self.enc_sizes_2d[::-1])

        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = self.curr_lstt_output
        
        if self.long_term_memories is None:
            self.long_term_memories = lstt_long_memories
        else:
            self.update_long_term_memory(lstt_long_memories,is_ref=True)
        self.ref_frame_num += 1
        self.last_mem_step = frame_step

        self.short_term_memories_list = [lstt_short_memories]
        self.short_term_memories = lstt_short_memories

    def set_prev_frame(self, img=None, mask=None, frame_step=1):
        self.frame_step = frame_step
        curr_enc_embs, curr_stuff_one_hot, curr_thing_one_hot = self.encode_one_img_mask(
            img, mask, frame_step)

        if curr_enc_embs is None:
            print('No image for previous frame!')
            exit()

        if curr_stuff_one_hot is None or curr_thing_one_hot is None:
            print('No mask for previous frame!')
            exit()

        self.curr_enc_embs = curr_enc_embs
        self.curr_stuff_one_hot = curr_stuff_one_hot
        self.curr_thing_one_hot = curr_thing_one_hot

        curr_id_embs = self.assign_identity(curr_stuff_one_hot, curr_thing_one_hot)
        self.curr_id_embs = curr_id_embs

        # self matching and propagation
        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      None,
                                                      None,
                                                      curr_id_embs,
                                                      pos_embs=self.pos_embs,
                                                      sizes_2d=self.enc_sizes_2d[::-1])

        lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories = self.curr_lstt_output

        if self.long_term_memories is None:
            self.long_term_memories = lstt_long_memories
        else:
            self.update_long_term_memory(lstt_long_memories)
        self.last_mem_step = frame_step

        self.short_term_memories_list = [lstt_short_memories]
        self.short_term_memories = lstt_short_memories

    def update_long_term_memory(self, new_long_term_memories, is_ref=False):
        updated_long_term_memories = []
        for new_long_term_memory, last_long_term_memory in zip(
                new_long_term_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(new_long_term_memory,
                                     last_long_term_memory):
                if not self.training:
                    e_len = new_e.shape[0]
                    e_num = last_e.shape[0] // e_len
                    max_num = self.cfg.TEST_LONG_TERM_MEM_MAX
                    if max_num <= e_num:
                        last_e = torch.cat([last_e[:e_len*(max_num-(self.ref_frame_num+1))],
                                            last_e[-self.ref_frame_num*e_len:]],dim=0)
                    if is_ref:
                        updated_e.append(torch.cat([last_e,new_e], dim=0))
                    else:
                        updated_e.append(torch.cat([new_e, last_e], dim=0))
                else:
                    updated_e.append(torch.cat([new_e, last_e], dim=0))
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories

    def update_short_term_memory(self, curr_mask, curr_id_embs=None):
        if curr_id_embs is None:
            if len(curr_mask.size()) == 3 or curr_mask.size()[0] == 1:
                stuff_mask,thing_mask = split_stuff_thing_mask(curr_mask,self.obj_mapping)
                curr_stuff_one_hot = one_hot_mask(stuff_mask, self.max_stuff_num, add_bk=False)
                curr_thing_one_hot = one_hot_mask(thing_mask, self.max_thing_num, add_bk=False)
            else:
                raise ValueError("mask shape error")
            curr_id_embs = self.assign_identity(curr_stuff_one_hot, curr_thing_one_hot)

        lstt_curr_memories = self.curr_lstt_output[1]
        lstt_curr_memories_2d = []
        if self.cfg.TRAIN_MS_LSTT_MEMORY_DILATION:
            dilated_lstt_curr_memories = []

        layer_idx = 0
        for s in range(len(self.curr_id_embs)):
            for i in range(self.cfg.MODEL_MS_LSTT_NUMS[s]):
                curr_k, curr_v = lstt_curr_memories[layer_idx][
                    0], lstt_curr_memories[layer_idx][1]
                curr_k, curr_v = self.AOT.MSLSTT.layers_list[s][i].fuse_key_value_id(
                    curr_k, curr_v, curr_id_embs[s])
                if self.cfg.MODEL_MS_ATT_DIMS[s] is not None:
                    curr_k = self.AOT.MSLSTT.layers_list[s][i].linear_Kd(curr_k)
                lstt_curr_memories[layer_idx][0], lstt_curr_memories[layer_idx][
                    1] = curr_k, curr_v
                lstt_curr_memories_2d.append([
                    seq_to_2d(lstt_curr_memories[layer_idx][0], self.enc_sizes_2d[-(s+1)]),
                    seq_to_2d(lstt_curr_memories[layer_idx][1], self.enc_sizes_2d[-(s+1)])
                ])
                if self.cfg.TRAIN_MS_LSTT_MEMORY_DILATION:
                    # update dilated long-term memory
                    if self.cfg.MODEL_MS_GLOBAL_DILATIONS[s] >1:
                        d = self.cfg.MODEL_MS_GLOBAL_DILATIONS[s]
                        local_k,local_v = lstt_curr_memories_2d[-1]
                        bs,ck,h,w = local_k.shape
                        bs,cv,h,w = local_v.shape
                        if self.cfg.MODEL_MS_CONV_DILATION:
                            dilated_k = self.AOT.MSLSTT.layers_list[s][i].dilation_conv_K(local_k).reshape(bs,ck,-1).permute(2,0,1)
                            dilated_v = self.AOT.MSLSTT.layers_list[s][i].dilation_conv_V(local_v).reshape(bs,cv,-1).permute(2,0,1)
                        else:
                            dilated_k = local_k[:,:,::d,::d].reshape(bs,ck,-1).permute(2,0,1)
                            dilated_v = local_v[:,:,::d,::d].reshape(bs,cv,-1).permute(2,0,1)
                        dilated_lstt_curr_memories.append([dilated_k,dilated_v])
                    else:
                        dilated_lstt_curr_memories.append(lstt_curr_memories[layer_idx])
                layer_idx += 1

        self.short_term_memories_list.append(lstt_curr_memories_2d)
        self.short_term_memories_list = self.short_term_memories_list[
            -self.short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]

        if self.frame_step - self.last_mem_step >= self.long_term_mem_gap:
            if self.cfg.TRAIN_MS_LSTT_MEMORY_DILATION:
                self.update_long_term_memory(dilated_lstt_curr_memories)
            else:
                self.update_long_term_memory(lstt_curr_memories)
            self.last_mem_step = self.frame_step

    def match_propogate_one_frame(self, img=None, img_embs=None):
        self.frame_step += 1
        if img_embs is None:
            curr_enc_embs, _,_ = self.encode_one_img_mask(
                img, None, self.frame_step)
        else:
            curr_enc_embs = img_embs
        self.curr_enc_embs = curr_enc_embs

        self.curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs,
                                                      self.long_term_memories,
                                                      self.short_term_memories,
                                                      None,
                                                      pos_embs=self.pos_embs,
                                                      sizes_2d=self.enc_sizes_2d[::-1])

    def decode_current_logits(self, output_size=None,intermediate_pred=False):

        curr_lstt_embs = self.curr_lstt_output[0]
        num_med = self.cfg.MODEL_MS_LSTT_NUMS[0]
        pred_id_logits_list = [self.AOT.decode_id_logits(curr_lstt_embs)]
        if intermediate_pred:
            pred_id_logits_list.append(self.AOT.decode_med_logits(curr_lstt_embs[:num_med+1],self.curr_enc_embs))
        
        for i in range(len(pred_id_logits_list)):
            if self.enable_id_shuffle:  # reverse shuffle
                pred_id_logits_list[i][:,:self.max_stuff_num,:,:] = torch.einsum('bohw,bto->bthw', pred_id_logits_list[i][:,:self.max_stuff_num,:,:],
                                            self.id_shuffle_matrix_stuff)
                pred_id_logits_list[i][:,self.max_stuff_num:,:,:] = torch.einsum('bohw,bto->bthw', 
                                            pred_id_logits_list[i][:,self.max_stuff_num:,:,:],
                                            self.id_shuffle_matrix_thing)

            # remove unused identities
            for b in range(len(self.obj_nums)):
                stuff_num = self.obj_nums[b][0]
                thing_num = self.obj_nums[b][1]
                pred_id_logits_list[i][b, stuff_num:self.max_stuff_num] = - \
                    1e+10 if pred_id_logits_list[i].dtype == torch.float32 else -1e+4
                pred_id_logits_list[i][b, self.max_stuff_num+thing_num:] = - \
                    1e+10 if pred_id_logits_list[i].dtype == torch.float32 else -1e+4
            
            pred_id_logits_list[i] = torch.einsum('bohw,bto->bthw', pred_id_logits_list[i],self.id_rearrange_matrix.permute(0,2,1))

        self.pred_id_logits_list = pred_id_logits_list

        if intermediate_pred:
            pred_id_logits = pred_id_logits_list[-1]
        else:
            pred_id_logits = pred_id_logits_list[0]
        if output_size is not None:
            for i in range(len(pred_id_logits_list)):
                pred_id_logits = F.interpolate(pred_id_logits,
                                            size=output_size,
                                            mode="bilinear",
                                            align_corners=self.align_corners)
        return pred_id_logits

    def predict_current_mask(self, output_size=None, return_prob=False):
        if output_size is None:
            output_size = self.input_size_2d
        
        pred_id_logits = self.pred_id_logits_list[0]
        pred_id_logits = F.interpolate(pred_id_logits,
                                       size=output_size,
                                       mode="bilinear",
                                       align_corners=self.align_corners)
        pred_mask = torch.argmax(pred_id_logits, dim=1)

        if not return_prob:
            return pred_mask
        else:
            pred_prob = torch.softmax(pred_id_logits, dim=1)
            return pred_mask, pred_prob

    def calculate_current_loss(self, gt_mask, step):
        if self.cfg.TRAIN_INTERMEDIATE_PRED_LOSS:
            pred_id_logits_list = [pred for pred in self.pred_id_logits_list]
        else:
            pred_id_logits_list = [self.pred_id_logits_list[0]]
        
        for i in range(len(pred_id_logits_list)):
            pred_id_logits_list[i] = F.interpolate(pred_id_logits_list[i],
                                        size=gt_mask.size()[-2:],
                                        mode="bilinear",
                                        align_corners=self.align_corners)

        total_loss = 0
        for i in range(len(pred_id_logits_list)):
            label_list = []
            logit_list = []
            for b in range(len(self.obj_nums)):
                stuff_num,thing_num = self.obj_nums[b][0],self.obj_nums[b][1]
                now_label = gt_mask[b].long()
                now_logit = pred_id_logits_list[i][b, :(stuff_num+thing_num)].unsqueeze(0)
                label_list.append(now_label.long())
                logit_list.append(now_logit)

            
            for loss, loss_weight in zip(self.losses, self.loss_weights):
                total_loss = total_loss + loss_weight * \
                    loss(logit_list, label_list, step)

        return total_loss

    def generate_loss_mask(self, gt_mask, step, return_prob=False):
        self.decode_current_logits(intermediate_pred=self.cfg.TRAIN_INTERMEDIATE_PRED_LOSS)
        loss = self.calculate_current_loss(gt_mask, step)
        if return_prob:
            mask, prob = self.predict_current_mask(return_prob=True)
            return loss, mask, prob
        else:
            mask = self.predict_current_mask()
            return loss, mask

    def keep_gt_mask(self, pred_mask, keep_prob=0.2):
        pred_mask = pred_mask.float()
        gt_mask = self.offline_masks[self.frame_step].float().squeeze(1)

        shape = [1 for _ in range(pred_mask.ndim)]
        shape[0] = self.batch_size
        random_tensor = keep_prob + torch.rand(
            shape, dtype=pred_mask.dtype, device=pred_mask.device)
        random_tensor.floor_()  # binarize

        pred_mask = pred_mask * (1 - random_tensor) + gt_mask * random_tensor

        return pred_mask

    def generate_split_matrix(self):
        rearrange = []
        for b in range(self.batch_size):
            default_range = [i for i in range(self.max_thing_num+self.max_stuff_num)]
            new_range = []
            stuff_idx = []
            thing_idx = []
            for k,v in self.obj_mapping[b].items():
                if v[0] == 0: #stuff
                    stuff_idx.append(k)
                else:
                    thing_idx.append(k)
                default_range.remove(k)
            split_num = self.max_stuff_num - len(stuff_idx)
            new_range = stuff_idx + default_range[:split_num] + thing_idx + default_range[split_num:]
            rearrange.append(new_range)
        self.id_rearrange_matrix = generate_rearrange_matrix(self.max_thing_num+self.max_stuff_num,self.batch_size,rearrange,self.gpu_id)
            
                
    def restart_engine(self, batch_size=1, enable_id_shuffle=False):

        self.batch_size = batch_size
        self.frame_step = 0
        self.last_mem_step = -1
        self.enable_id_shuffle = enable_id_shuffle
        self.freeze_id = False

        self.obj_nums = None
        self.pos_embs = None
        self.enc_sizes_2d = []
        self.enc_hws = []
        self.input_size_2d = None

        self.long_term_memories = None
        self.ref_frame_num = 0
        self.short_term_memories_list = []
        self.short_term_memories = None

        self.enable_offline_enc = False
        self.offline_enc_embs = None
        self.offline_one_hot_masks = None
        self.offline_frames = -1
        self.total_offline_frame_num = 0

        self.curr_enc_embs = None
        self.curr_memories = None
        self.curr_id_embs = None

        if enable_id_shuffle:
            self.id_shuffle_matrix_stuff = generate_permute_matrix(
                self.max_stuff_num, batch_size, gpu_id=self.gpu_id)
            self.id_shuffle_matrix_thing = generate_permute_matrix(
                self.max_thing_num, batch_size, gpu_id=self.gpu_id)
        else:
            self.id_shuffle_matrix = None

    def update_sizes(self, input_size, enc_sizes):
        self.input_size_2d = input_size
        self.enc_sizes_2d = enc_sizes
        self.enc_hws = [size[0]*size[1] for size in enc_sizes]


class PAOTInferEngine(nn.Module):
    def __init__(self,
                 aot_model,
                 gpu_id=0,
                 long_term_mem_gap=9999,
                 short_term_mem_skip=1,
                 max_aot_obj_num=None):
        super().__init__()

        self.cfg = aot_model.cfg
        self.AOT = aot_model

        self.max_stuff_num = aot_model.max_stuff_num
        self.max_thing_num = aot_model.max_thing_num

        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        from typing import List
        self.aot_engines = [] # type: List[PAOTEngine]

        self.restart_engine()

    def restart_engine(self):
        self.aot_engines = []
        self.engines_stuff = []
        self.engines_thing = []
        self.engines_obj_nums = []
        self.engines_obj_mapping = []

        self.total_obj = []
        self.obj_mapping = {}
    
    def renew_engine(self):
        self.engines_stuff = []
        self.engines_thing = []
        self.engines_obj_nums = []
        self.engines_obj_mapping = []
    
    def separate_reference_mask(self, mask): 
        # record obj index for each engine
        self.total_obj = list(self.obj_mapping.keys())
        
        self.renew_engine()
        engine_thing = []
        engine_stuff = [0]
        for obj in self.total_obj:
            if obj == 0:
                continue
            elif self.obj_mapping[obj][0] == 0:
                engine_stuff.append(obj)
            elif self.obj_mapping[obj][0] == 1:
                engine_thing.append(obj)
            if len(engine_stuff) == self.max_stuff_num:
                self.engines_stuff.append(engine_stuff)
                engine_stuff = [0]
            if len(engine_thing) == self.max_thing_num:
                self.engines_thing.append(engine_thing)
                engine_thing = []
        # none exceed max num
        if engine_stuff != [0]:
            self.engines_stuff.append(engine_stuff)
        if engine_thing != []:
            self.engines_thing.append(engine_thing)
        # align length
        aot_num = max(len(self.engines_stuff),len(self.engines_thing))
        for i in range(aot_num - len(self.engines_stuff)):
            self.engines_stuff.append([0])
        for i in range(aot_num - len(self.engines_thing)):
            self.engines_thing.append([])
        
        # separate mask, each mask value from 0~max_obj
        separated_masks = []
        for i in range(aot_num):
            idx = 0
            engine_obj_mapping = {}
            sep_mask = torch.zeros_like(mask)
            for l,j in enumerate(self.engines_stuff[i]):
                sep_mask += ((mask==j) * idx)
                engine_obj_mapping[idx] = [0,l]
                idx += 1
                
            for l,k in enumerate(self.engines_thing[i]):
                sep_mask += ((mask==k) * idx)
                engine_obj_mapping[idx] = [1,l]
                idx += 1
                
            separated_masks.append(sep_mask)
            self.engines_obj_mapping.append([engine_obj_mapping])
            self.engines_obj_nums.append([[len(self.engines_stuff[i]),len(self.engines_thing[i])]])
        return separated_masks
        
    def separate_pred_mask(self, mask):
        # separate mask, each mask value from 0~max_obj
        separated_masks = []
        for i in range(len(self.aot_engines)):
            idx = 0
            sep_mask = torch.zeros_like(mask)
            for j in self.engines_stuff[i]:
                sep_mask += ((mask==j) * idx)
                idx += 1
                
            for k in self.engines_thing[i]:
                sep_mask += ((mask==k) * idx)
                idx += 1
                
            separated_masks.append(sep_mask)
        return separated_masks
    
    def min_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_logits = []
        bg_logits = []

        for logit in all_logits:
            bg_logits.append(logit[:, 0:1])
            fg_logits.append(logit[:, 1:1 + self.max_aot_obj_num])

        bg_logit, _ = torch.min(torch.cat(bg_logits, dim=1),
                                dim=1,
                                keepdim=True)
        merged_logit = torch.cat([bg_logit] + fg_logits, dim=1)

        return merged_logit
    
    def soft_logit_aggregation(self, all_logits):
        h,w = all_logits[0].shape[2:]
        merged_logit = torch.zeros((1,len(self.total_obj),h,w),device=all_logits[0].device)
        bg_probs = []
        # print(self.total_obj,len(self.total_obj))
        # print(self.engines_stuff,len(self.engines_stuff))
        # print(self.engines_thing,len(self.engines_thing))
        # print('\n')
        for i,logits in enumerate(all_logits):
            prob = F.softmax(logits,dim=1)
            bg_probs.append(prob[:,0:1])
            idx = 0
            for j in self.engines_stuff[i]:
                merged_logit[:,j] = prob[:,idx]
                idx += 1
            for j in self.engines_thing[i]:
                merged_logit[:,j] = prob[:,idx]
                idx += 1
        bg_prob = torch.prod(torch.cat(bg_probs, dim=1), dim=1, keepdim=True)
        merged_logit[:,0:1] = bg_prob
        merged_logit = torch.logit(merged_logit.clamp(1e-5, 1 - 1e-5))
        return merged_logit

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        obj_mapping = obj_nums
        self.obj_mapping.update(obj_mapping)
        separated_masks = self.separate_reference_mask(mask)
        while (len(separated_masks) > len(self.aot_engines)):
            new_engine = PAOTEngine(self.AOT, self.gpu_id,
                                     self.long_term_mem_gap,
                                     self.short_term_mem_skip)
            new_engine.eval()
            self.aot_engines.append(new_engine)
        
        img_embs = None
        for i,aot_engine in enumerate(self.aot_engines):
            aot_engine.obj_mapping = self.engines_obj_mapping[i]
            aot_engine.add_reference_frame(img,
                                           separated_masks[i],
                                           obj_nums=self.engines_obj_nums[i],
                                           frame_step=frame_step,
                                           img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs
        
        self.update_sizes()

    def match_propogate_one_frame(self, img=None):
        img_embs = None
        for aot_engine in self.aot_engines:
            aot_engine.match_propogate_one_frame(img, img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = aot_engine.curr_enc_embs

    def decode_current_logits(self, output_size=None):
        all_logits = []
        for aot_engine in self.aot_engines:
            all_logits.append(aot_engine.decode_current_logits(output_size,
                intermediate_pred=self.cfg.TEST_INTERMEDIATE_PRED))
        pred_id_logits = self.soft_logit_aggregation(all_logits)
        return pred_id_logits

    def update_memory(self, curr_mask):
        separated_masks = self.separate_pred_mask(curr_mask)
        for aot_engine, separated_mask in zip(self.aot_engines,
                                              separated_masks):
            aot_engine.update_short_term_memory(separated_mask)

    def update_sizes(self):
        self.input_size_2d = self.aot_engines[0].input_size_2d
        self.enc_sizes_2d = self.aot_engines[0].enc_sizes_2d
        self.enc_hws = self.aot_engines[0].enc_hws

