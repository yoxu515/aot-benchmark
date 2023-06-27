from tkinter.messagebox import NO
from numpy import size
import torch
import torch.nn.functional as F
from torch import nn

from networks.layers.attention import MultiheadAttention, MultiheadLocalAttentionV2, MultiheadLocalAttentionV3,silu, GAU, LocalGAU, GatedPropagation, LocalGatedPropagation
from networks.layers.basic import ConvGN,ResGN,DropPath, GroupNorm1D, GNActDWConv2d, seq_to_2d, ScaleOffset, mask_out

def _get_norm(indim, type='ln', groups=8):
    if type == 'gn':
        return GroupNorm1D(indim, groups)
    else:
        return nn.LayerNorm(indim)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(
        F"activation should be relu/gele/glu, not {activation}.")


class LongShortTermTransformer(nn.Module):
    def __init__(self,
                 num_layers=2,
                 d_model=256,
                 self_nhead=8,
                 att_nhead=8,
                 dim_feedforward=1024,
                 emb_dropout=0.,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 droppath_scaling=False,
                 activation="gelu",
                 return_intermediate=False,
                 intermediate_norm=True,
                 final_norm=True,
                 block_version="v1"):

        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.emb_dropout = nn.Dropout(emb_dropout, True)

        if block_version == "v1":
            block = LongShortTermTransformerBlock
        elif block_version == "v2":
            block = LongShortTermTransformerBlockV2
        else:
            raise NotImplementedError

        layers = []
        for idx in range(num_layers):
            if droppath_scaling:
                if num_layers == 1:
                    droppath_rate = 0
                else:
                    droppath_rate = droppath * idx / (num_layers - 1)
            else:
                droppath_rate = droppath
            layers.append(
                block(d_model, self_nhead, att_nhead, dim_feedforward,
                      droppath_rate, lt_dropout, st_dropout, droppath_lst,
                      activation))
        self.layers = nn.ModuleList(layers)

        num_norms = num_layers - 1 if intermediate_norm else 0
        if final_norm:
            num_norms += 1
        self.decoder_norms = [
            _get_norm(d_model, type='ln') for _ in range(num_norms)
        ] if num_norms > 0 else None

        if self.decoder_norms is not None:
            self.decoder_norms = nn.ModuleList(self.decoder_norms)

    def forward(self,
                tgt,
                long_term_memories,
                short_term_memories,
                curr_id_emb=None,
                self_pos=None,
                size_2d=None):

        output = self.emb_dropout(tgt)

        intermediate = []
        intermediate_memories = []

        for idx, layer in enumerate(self.layers):
            output, memories = layer(output,
                                     long_term_memories[idx] if
                                     long_term_memories is not None else None,
                                     short_term_memories[idx] if
                                     short_term_memories is not None else None,
                                     curr_id_emb=curr_id_emb,
                                     self_pos=self_pos,
                                     size_2d=size_2d)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_memories.append(memories)

        if self.decoder_norms is not None:
            if self.final_norm:
                output = self.decoder_norms[-1](output)

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

                if self.intermediate_norm:
                    for idx in range(len(intermediate) - 1):
                        intermediate[idx] = self.decoder_norms[idx](
                            intermediate[idx])

        if self.return_intermediate:
            return intermediate, intermediate_memories

        return output, memories


class SharedLongShortTermTransformer(nn.Module):
    def __init__(self,
                 num_layers=2,
                 d_model=256,
                 self_nhead=8,
                 att_nhead=8,
                 dim_feedforward=1024,
                 emb_dropout=0.,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 droppath_scaling=False,
                 activation="gelu",
                 return_intermediate=False,
                 intermediate_norm=True,
                 final_norm=True,
                 block_version="v1"):

        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.emb_dropout = nn.Dropout(emb_dropout, True)

        if block_version == "v1":
            block = LongShortTermTransformerBlock
        elif block_version == "v2":
            block = LongShortTermTransformerBlockV2
        else:
            raise NotImplementedError

        self.layer = block(d_model, self_nhead, att_nhead, dim_feedforward,
                           droppath, lt_dropout, st_dropout, droppath_lst,
                           activation)

        self.decoder_norm = _get_norm(d_model, type='ln')

    def forward(self,
                tgt,
                long_term_memories,
                short_term_memories,
                curr_id_emb=None,
                self_pos=None,
                size_2d=None):

        # output = self.emb_dropout(tgt)
        output = tgt

        intermediate = []
        intermediate_memories = []

        for idx in range(self.num_layers):
            output, memories = self.layer(
                output,
                long_term_memories[idx]
                if long_term_memories is not None else None,
                short_term_memories[idx]
                if short_term_memories is not None else None,
                curr_id_emb=curr_id_emb,
                self_pos=self_pos,
                size_2d=size_2d)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_memories.append(memories)

        if self.final_norm:
            output = self.decoder_norm(output)

        if self.return_intermediate:
            intermediate.pop()
            intermediate.append(output)

            if self.intermediate_norm:
                for idx in range(len(intermediate) - 1):
                    intermediate[idx] = self.decoder_norm(intermediate[idx])

        if self.return_intermediate:
            return intermediate, intermediate_memories

        return output, memories


class LongShortTermTransformerBlock(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 activation="gelu",
                 local_dilation=1,
                 enable_corr=True):
        super().__init__()

        # Long Short-Term Attention
        self.norm1 = _get_norm(d_model)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)

        self.long_term_attn = MultiheadAttention(d_model,
                                                 att_nhead,
                                                 use_linear=False,
                                                 dropout=lt_dropout)

        MultiheadLocalAttention = MultiheadLocalAttentionV2 if enable_corr else MultiheadLocalAttentionV3
        self.short_term_attn = MultiheadLocalAttention(d_model,
                                                       att_nhead,
                                                       dilation=local_dilation,
                                                       use_linear=False,
                                                       dropout=st_dropout)
        self.lst_dropout = nn.Dropout(max(lt_dropout, st_dropout), True)
        self.droppath_lst = droppath_lst

        # Self-attention
        self.norm2 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30)):

        # Self-attention
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_Q = self.linear_Q(_tgt)
        curr_K = curr_Q
        curr_V = _tgt

        local_Q = seq_to_2d(curr_Q, size_2d)

        if curr_id_emb is not None:
            global_K, global_V = self.fuse_key_value_id(
                curr_K, curr_V, curr_id_emb)
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory

        tgt2 = self.long_term_attn(curr_Q, global_K, global_V)[0]
        tgt3 = self.short_term_attn(local_Q, local_K, local_V)[0]

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3)

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [[curr_K, curr_V], [global_K, global_V],
                     [local_K, local_V]]

    def fuse_key_value_id(self, key, value, id_emb):
        K = key
        V = self.linear_V(value + id_emb)
        return K, V

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class LongShortTermTransformerBlockV2(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 activation="gelu",
                 local_dilation=1,
                 enable_corr=True):
        super().__init__()
        self.d_model = d_model
        self.att_nhead = att_nhead

        # Self-attention
        self.norm1 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Long Short-Term Attention
        self.norm2 = _get_norm(d_model)
        self.linear_QV = nn.Linear(d_model, 2 * d_model)
        self.linear_ID_KV = nn.Linear(d_model, d_model + att_nhead)

        self.long_term_attn = MultiheadAttention(d_model,
                                                 att_nhead,
                                                 use_linear=False,
                                                 dropout=lt_dropout)

        MultiheadLocalAttention = MultiheadLocalAttentionV2 if enable_corr else MultiheadLocalAttentionV3
        self.short_term_attn = MultiheadLocalAttention(d_model,
                                                       att_nhead,
                                                       dilation=local_dilation,
                                                       use_linear=False,
                                                       dropout=st_dropout)
        self.lst_dropout = nn.Dropout(max(lt_dropout, st_dropout), True)
        self.droppath_lst = droppath_lst

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30)):

        # Self-attention
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_QV = self.linear_QV(_tgt)
        curr_QV = torch.split(curr_QV, self.d_model, dim=2)
        curr_Q = curr_K = curr_QV[0]
        curr_V = curr_QV[1]

        local_Q = seq_to_2d(curr_Q, size_2d)

        if curr_id_emb is not None:
            global_K, global_V = self.fuse_key_value_id(
                curr_K, curr_V, curr_id_emb)

            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory

        tgt2 = self.long_term_attn(curr_Q, global_K, global_V)[0]
        tgt3 = self.short_term_attn(local_Q, local_K, local_V)[0]

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3)

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [[curr_K, curr_V], [global_K, global_V],
                     [local_K, local_V]]

    def fuse_key_value_id(self, key, value, id_emb):
        ID_KV = self.linear_ID_KV(id_emb)
        ID_K, ID_V = torch.split(ID_KV, [self.att_nhead, self.d_model], dim=2)
        bs = key.size(1)
        K = key.view(-1, bs, self.att_nhead, self.d_model //
                     self.att_nhead) * (1 + torch.tanh(ID_K)).unsqueeze(-1)
        K = K.view(-1, bs, self.d_model)
        V = value + ID_V
        return K, V

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class MSLongShortTermTransformer(nn.Module):
    def __init__(self,
                 num_layers=[3,1,1,1],
                 d_encoder=[24, 32, 96, 1280],
                 d_model=[256,128,64,32],
                 self_nheads=[8,4,1,1],
                 att_nheads=[8,4,1,1],
                 dims_feedforward=[1024,512,256,128],
                 global_dilations=[1,1,2,4],
                 local_dilations=[1,1,1,1],
                 memory_dilation=False,
                 conv_dilation=False,
                 att_dims=[None,None,None,None],
                 emb_dropouts=[0.,0.,0.,0.],
                 droppath=[0.1,0.1,0.1,0.1],
                 lt_dropout=[0.,0.,0.,0.],
                 st_dropout=[0.,0.,0.,0.],
                 droppath_lst=[False,False,False,False],
                 droppath_scaling=[False,False,False,False],
                 activation="gelu",
                 return_intermediate=False,
                 intermediate_norm=True,
                 final_norm=True,
                 align_corners=True,
                 decode_intermediate_input=False,
                 decoder_res=False,
                 decoder_res_in=False,
                 use_relative_v=True,
                 use_self_pos=True,
                 topk=-1):
        super().__init__()
        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.align_corners = align_corners
        self.decode_intermediate_input = decode_intermediate_input
        
        self.emb_dropouts = nn.ModuleList()
        for i in range(len(d_model)):
            self.emb_dropouts.append(nn.Dropout(emb_dropouts[i], True))
        
        # LSTT layers
        block = MSLongShortTermTransformerBlock

        self.layers_list = nn.ModuleList()
        layer_idx = 0
        for s,num in enumerate(num_layers):
            layers = nn.ModuleList()
            for idx in range(num):
                if droppath_scaling[s]:
                    if num == 0 or num == 1:
                        droppath_rate = 0
                    else:
                        droppath_rate = droppath[s] * idx / (num - 1)
                else:
                    droppath_rate = droppath[s]
                
                layers.append(
                    block(d_model[s], self_nheads[s], att_nheads[s], dims_feedforward[s],
                        droppath_rate, lt_dropout[s], st_dropout[s], droppath_lst[s],
                        activation,global_dilation=global_dilations[s],
                        local_dilation=local_dilations[s],memory_dilation=memory_dilation,
                        d_att=att_dims[s],conv_dilation=conv_dilation,
                        use_relative_v=use_relative_v,use_self_pos=use_self_pos,
                        topk=topk if layer_idx==0 else -1))
                layer_idx += 1
            self.layers_list.append(layers)
        
        # decoder layers
        self.decoder_norms = nn.ModuleList()
        self.convs_in = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.convs_out = nn.ModuleList()
        
        for s,num in enumerate(num_layers):
            for i in range(num):
                if self.intermediate_norm:
                    self.decoder_norms.append(_get_norm(d_model[s], type='ln'))
                else:
                    self.decoder_norms.append(nn.Identity())
            if num >0 and not final_norm:
                self.decoder_norms[-1] = nn.Identity()
        
        for s in range(len(num_layers)-1):
            if self.decode_intermediate_input:
                if decoder_res_in:
                    self.convs_in.append(ResGN(d_model[s]*(num_layers[s]+1), d_model[s+1]))
                else:
                    self.convs_in.append(ConvGN(d_model[s]*(num_layers[s]+1), d_model[s+1],1))
            else:
                if decoder_res_in:
                    self.convs_in.append(ResGN(d_model[s+1], d_model[s+1]))
                else:
                    self.convs_in.append(ConvGN(d_model[s], d_model[s+1], 1))
            if decoder_res:
                self.convs_out.append(ResGN(d_model[s+1], d_model[s+1]))
            else:
                self.convs_out.append(ConvGN(d_model[s+1], d_model[s+1], 3))

    def forward(self,
                embs,
                long_term_memories,
                short_term_memories,
                curr_id_embs=None,
                self_pos=None,
                sizes_2d=None):
        
        
        embs = list(reversed(embs))
        output = embs[0]
        bs, c, h, w = output.size()
        output = output.view(bs, c, h * w).permute(2, 0, 1) # (B,C,H,W) -> (HW,B,C)
        all_outputs = []
        all_memories = []
        tmp_outputs = [output]
        tmp_memories = []
        
        idx = 0
        s=0
        for layer in self.layers_list[s]:
            output, memories = layer(output,
                                        long_term_memories[idx] if
                                            long_term_memories is not None else None,
                                        short_term_memories[idx] if
                                            short_term_memories is not None else None,
                                        curr_id_emb=curr_id_embs[s] if
                                            curr_id_embs is not None else None,
                                        self_pos=self_pos[s] if 
                                            self_pos is not None else None,
                                        size_2d=sizes_2d[s])
            # decoder norm
            if self.decoder_norms is not None:
                output = self.decoder_norms[idx](output)
            
            tmp_outputs.append(output)
            tmp_memories.append(memories)
            idx += 1
        if self.return_intermediate:
            all_outputs = all_outputs + tmp_outputs
            all_memories = all_memories + tmp_memories
        
        for layers in self.layers_list[1:]: # loop in scale
            if s==0 and len(tmp_outputs) == 1: # skip first scale if layer is 0
                x = embs[s+1]
            else:
                # merge lstt layer outputs
                if self.decode_intermediate_input:
                    for i in range(len(tmp_outputs)):
                        tmp_outputs[i] = tmp_outputs[i].view(sizes_2d[s][0],sizes_2d[s][1],bs,-1).permute(2,3,0,1) #(HW,B,C) -> (B,C,H,W)
                    x = torch.cat(tmp_outputs, dim=1)
                else:
                    x = tmp_outputs[-1].view(sizes_2d[s][0],sizes_2d[s][1],bs,-1).permute(2,3,0,1)
                # down channel
                x = F.relu(self.convs_in[s](x)) 
                # upscale
                x = F.interpolate(x,
                            size=sizes_2d[s+1],
                            mode="bilinear",
                            align_corners=self.align_corners)
                # add next scale feature
                x = F.relu(self.convs_out[s](embs[s+1] + x))
            
            # input to next scale
            s += 1
            output = x

            bs, c, h, w = output.size()
            output = output.view(bs, c, h * w).permute(2, 0, 1) # (B,C,H,W) -> (HW,B,C)
            tmp_outputs = [output]
            tmp_memories = []

            for layer in layers: # loop in LSTT layer
                output, memories = layer(output,
                                        long_term_memories[idx] if
                                            long_term_memories is not None else None,
                                        short_term_memories[idx] if
                                            short_term_memories is not None else None,
                                        curr_id_emb=curr_id_embs[s] if
                                            curr_id_embs is not None else None,
                                        self_pos=self_pos[s] if
                                            self_pos is not None else None,
                                        size_2d=sizes_2d[s])
                # decoder norm
                if self.decoder_norms is not None:
                    output = self.decoder_norms[idx](output)
                
                tmp_outputs.append(output)
                tmp_memories.append(memories)
                idx += 1

            if self.return_intermediate:
                all_outputs = all_outputs + tmp_outputs
                all_memories = all_memories + tmp_memories
                    
                
        
        if self.decode_intermediate_input:
            for i in range(len(tmp_outputs)):
                tmp_outputs[i] = tmp_outputs[i].view(sizes_2d[s][0],sizes_2d[s][1],bs,-1).permute(2,3,0,1) #(B,C,H,W)
            output = torch.cat(tmp_outputs, dim=1)
        else:
            output = tmp_outputs[-1].view(sizes_2d[s][0],sizes_2d[s][1],bs,-1).permute(2,3,0,1)
        all_outputs.append(output)

        if self.return_intermediate:
            return all_outputs, all_memories

        return output, memories


class MSLongShortTermTransformerBlock(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 activation="gelu",
                 global_dilation=1,
                 local_dilation=1,
                 memory_dilation=False,
                 conv_dilation=False,
                 d_att=None,
                 enable_corr=True,
                 use_relative_v=True,
                 use_self_pos=True,
                 topk=-1):
        super().__init__()

        self.d_model = d_model
        self.att_nhead = att_nhead
        self.memory_dilation=memory_dilation
        self.use_self_pos = use_self_pos

        # Self-attention
        self.norm1 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Long Short-Term Attention
        self.global_dilation = global_dilation
        self.conv_dilation = conv_dilation
        if global_dilation>1 and conv_dilation:
            if d_att is not None:
                self.dilation_conv_K = nn.Conv2d(d_att,d_att,kernel_size=global_dilation,stride=global_dilation)
            else:
                self.dilation_conv_K = nn.Conv2d(d_model,d_model,kernel_size=global_dilation,stride=global_dilation)
            self.dilation_conv_V = nn.Conv2d(d_model,d_model,kernel_size=global_dilation,stride=global_dilation)
        self.d_att = d_att
        if d_att is not None:
            self.linear_Qd = nn.Linear(d_model,d_att)
            self.linear_Kd = nn.Linear(d_model,d_att)
        self.norm2 = _get_norm(d_model)
        self.linear_QV = nn.Linear(d_model, 2 * d_model)
        self.linear_ID_KV = nn.Linear(d_model, d_model + att_nhead)

        self.long_term_attn = MultiheadAttention(d_model,
                                                 att_nhead,
                                                 use_linear=False,
                                                 dropout=lt_dropout,
                                                 d_att=d_att,
                                                 top_k=topk)

        MultiheadLocalAttention = MultiheadLocalAttentionV2 if enable_corr else MultiheadLocalAttentionV3
        self.short_term_attn = MultiheadLocalAttention(d_model,
                                                       att_nhead,
                                                       dilation=local_dilation,
                                                       use_linear=False,
                                                       dropout=st_dropout,
                                                       d_att=d_att,
                                                       use_relative_v=use_relative_v)
        self.lst_dropout = nn.Dropout(max(lt_dropout, st_dropout), True)
        self.droppath_lst = droppath_lst

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30)):

        # Self-attention
        _tgt = self.norm1(tgt)
        if not self.use_self_pos:
            self_pos = None
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        if self.global_dilation > 1:
            k = k[::self.global_dilation,:,:]
            v = v[::self.global_dilation,:,:]
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_QV = self.linear_QV(_tgt)
        curr_QV = torch.split(curr_QV, self.d_model, dim=2)
        curr_Q = curr_K = curr_QV[0]
        curr_V = curr_QV[1]

        if self.d_att is not None:
            curr_Q = self.linear_Qd(curr_Q)
        local_Q = seq_to_2d(curr_Q, size_2d)

        if curr_id_emb is not None:
            global_K, global_V = self.fuse_key_value_id(
                curr_K, curr_V, curr_id_emb)
            if self.d_att is not None:
                global_K = self.linear_Kd(global_K)
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)

            if self.global_dilation>1 and self.memory_dilation:
                nhw,bs,ck = global_K.shape
                cv = global_V.shape[-1]
                # n = nhw // (size_2d[0] * size_2d[1])
                d = self.global_dilation
                if self.conv_dilation:
                    unfold_K = global_K.permute(1,2,0).reshape(bs,ck,size_2d[0],size_2d[1])
                    unfold_V = global_V.permute(1,2,0).reshape(bs,cv,size_2d[0],size_2d[1])
                    global_K = self.dilation_conv_K(unfold_K).reshape(bs,ck,-1).permute(2,0,1)
                    global_V = self.dilation_conv_V(unfold_V).reshape(bs,cv,-1).permute(2,0,1)
                else:
                    unfold_K = global_K.view(size_2d[0],size_2d[1],bs,ck)
                    unfold_V = global_V.view(size_2d[0],size_2d[1],bs,cv)
                    global_K = unfold_K[::d,::d,:,:].reshape(-1,bs,ck)
                    global_V = unfold_V[::d,::d,:,:].reshape(-1,bs,cv)
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory
        
        
        if self.memory_dilation:
            tgt2 = self.long_term_attn(curr_Q, global_K, global_V)[0]
        else:
            if self.global_dilation>1:
                nhw,bs,ck = global_K.shape
                cv = global_V.shape[-1]
                n = nhw // (size_2d[0] * size_2d[1])
                d = self.global_dilation
                if self.conv_dilation:
                    unfold_K = global_K.permute(1,2,0).reshape(bs*n,ck,size_2d[0],size_2d[1])
                    unfold_V = global_V.permute(1,2,0).reshape(bs*n,cv,size_2d[0],size_2d[1])
                    dilated_K = self.dilation_conv_K(unfold_K).reshape(bs,ck,-1).permute(2,0,1)
                    dilated_V = self.dilation_conv_V(unfold_V).reshape(bs,cv,-1).permute(2,0,1)
                else:
                    unfold_K = global_K.view(n,size_2d[0],size_2d[1],bs,ck)
                    unfold_V = global_V.view(n,size_2d[0],size_2d[1],bs,cv)
                    dilated_K = unfold_K[:,::d,::d,:,:].reshape(-1,bs,ck)
                    dilated_V = unfold_V[:,::d,::d,:,:].reshape(-1,bs,cv)
            else:
                dilated_K,dilated_V = global_K,global_V
            tgt2 = self.long_term_attn(curr_Q, dilated_K, dilated_V)[0]

        tgt3 = self.short_term_attn(local_Q, local_K, local_V)[0]

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3)

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [[curr_K, curr_V], [global_K, global_V],
                     [local_K, local_V]]

    def fuse_key_value_id(self, key, value, id_emb):
        ID_KV = self.linear_ID_KV(id_emb)
        ID_K, ID_V = torch.split(ID_KV, [self.att_nhead, self.d_model], dim=2)
        bs = key.size(1)
        K = key.view(-1, bs, self.att_nhead, self.d_model //
                     self.att_nhead) * (1 + torch.tanh(ID_K)).unsqueeze(-1)
        K = K.view(-1, bs, self.d_model)
        V = value + ID_V
        return K, V

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class FastLongShortTermTransformerBlock(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 activation="gelu",
                 local_dilation=1,
                 global_dilation=1,
                 enable_corr=True):
        super().__init__()

        self.d_model = d_model
        self.att_nhead = att_nhead

        # Self-attention
        self.norm1 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Long Short-Term Attention
        self.global_dilation = global_dilation
        self.norm2 = _get_norm(d_model)
        self.linear_QV = nn.Linear(d_model, 2 * d_model)
        self.linear_ID_KV = nn.Linear(d_model, d_model + att_nhead)

        self.long_term_attn = MultiheadAttention(d_model,
                                                 att_nhead,
                                                 use_linear=False,
                                                 dropout=lt_dropout)

        MultiheadLocalAttention = MultiheadLocalAttentionV2 if enable_corr else MultiheadLocalAttentionV3
        self.short_term_attn = MultiheadLocalAttention(d_model,
                                                       att_nhead,
                                                       dilation=local_dilation,
                                                       use_linear=False,
                                                       dropout=st_dropout,
                                                       return_V=True)
        self.lst_dropout = nn.Dropout(max(lt_dropout, st_dropout), True)
        self.droppath_lst = droppath_lst

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30)):

        # Self-attention
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        if self.global_dilation > 1:
            k = k[::self.global_dilation,:,:]
            v = v[::self.global_dilation,:,:]
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_QV = self.linear_QV(_tgt)
        curr_QV = torch.split(curr_QV, self.d_model, dim=2)
        curr_Q = curr_K = curr_QV[0]
        curr_V = curr_QV[1]

        local_Q = seq_to_2d(curr_Q, size_2d)

        if curr_id_emb is not None:
            global_K, global_V = self.fuse_key_value_id(
                curr_K, curr_V, curr_id_emb)
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory
        
        if self.global_dilation>1:
            nhw,bs,c = global_K.shape
            n = nhw // (size_2d[0] * size_2d[1])
            d = self.global_dilation
            unfold_K = global_K.view(n,size_2d[0],size_2d[1],bs,c)
            unfold_V = global_V.view(n,size_2d[0],size_2d[1],bs,c)
            dilated_K = unfold_K[:,::d,::d,:,:].reshape(-1,bs,c)
            dilated_V = unfold_V[:,::d,::d,:,:].reshape(-1,bs,c)
        else:
            dilated_K,dilated_V = global_K,global_V
        
        tgt3,qk = self.short_term_attn(local_Q, local_K, local_V)
        local_entropy = -torch.sum(F.softmax(qk)*F.log_softmax(qk,dim=2),dim=2) # B,C,HW
        
        tgt2 = self.long_term_attn(curr_Q, dilated_K, dilated_V)[0]
        

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3)

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [[curr_K, curr_V], [global_K, global_V],
                     [local_K, local_V]]

    def fuse_key_value_id(self, key, value, id_emb):
        ID_KV = self.linear_ID_KV(id_emb)
        ID_K, ID_V = torch.split(ID_KV, [self.att_nhead, self.d_model], dim=2)
        bs = key.size(1)
        K = key.view(-1, bs, self.att_nhead, self.d_model //
                     self.att_nhead) * (1 + torch.tanh(ID_K)).unsqueeze(-1)
        K = K.view(-1, bs, self.d_model)
        V = value + ID_V
        return K, V

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class DualBranchGPM(nn.Module):
    def __init__(self,
                 num_layers=2,
                 d_model=256,
                 self_nhead=8,
                 att_nhead=8,
                 dim_feedforward=1024,
                 emb_dropout=0.,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 droppath_scaling=False,
                 activation="gelu",
                 return_intermediate=False,
                 intermediate_norm=True,
                 final_norm=True,
                 topk=-1):

        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.emb_dropout = nn.Dropout(emb_dropout, True)
        # self.mask_token = nn.Parameter(torch.randn([1, 1, d_model]))

        block = GatedPropagationModule

        layers = []
        for idx in range(num_layers):
            if droppath_scaling:
                if num_layers == 1:
                    droppath_rate = 0
                else:
                    droppath_rate = droppath * idx / (num_layers - 1)
            else:
                droppath_rate = droppath
            layers.append(
                block(d_model,
                      self_nhead,
                      att_nhead,
                      dim_feedforward,
                      droppath_rate,
                      lt_dropout,
                      st_dropout,
                      droppath_lst,
                      activation,
                      layer_idx=idx,
                      topk=topk if idx==0 else -1))
        self.layers = nn.ModuleList(layers)

        num_norms = num_layers - 1 if intermediate_norm else 0
        if final_norm:
            num_norms += 1
        self.decoder_norms = [
            _get_norm(d_model * 2, type='gn', groups=2)
            for _ in range(num_norms)
        ] if num_norms > 0 else None

        if self.decoder_norms is not None:
            self.decoder_norms = nn.ModuleList(self.decoder_norms)

    def forward(self,
                tgt,
                long_term_memories,
                short_term_memories,
                curr_id_emb=None,
                self_pos=None,
                size_2d=None):

        output = self.emb_dropout(tgt)

        # output = mask_out(output, self.mask_token, 0.15, self.training)

        intermediate = []
        intermediate_memories = []
        output_id = None

        for idx, layer in enumerate(self.layers):
            output, output_id, memories = layer(
                output,
                output_id,
                long_term_memories[idx]
                if long_term_memories is not None else None,
                short_term_memories[idx]
                if short_term_memories is not None else None,
                curr_id_emb=curr_id_emb,
                self_pos=self_pos,
                size_2d=size_2d)

            cat_output = torch.cat([output, output_id], dim=2)

            if self.return_intermediate:
                intermediate.append(cat_output)
                intermediate_memories.append(memories)

        if self.decoder_norms is not None:
            if self.final_norm:
                cat_output = self.decoder_norms[-1](cat_output)

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(cat_output)

                if self.intermediate_norm:
                    for idx in range(len(intermediate) - 1):
                        intermediate[idx] = self.decoder_norms[idx](
                            intermediate[idx])

        if self.return_intermediate:
            return intermediate, intermediate_memories

        return cat_output, memories
class GatedPropagationModule(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 activation="gelu",
                 local_dilation=1,
                 enable_corr=True,
                 max_local_dis=7,
                 layer_idx=0,
                 expand_ratio=2.,
                 topk=-1):
        super().__init__()
        expand_ratio = expand_ratio
        expand_d_model = int(d_model * expand_ratio)
        self.expand_d_model = expand_d_model
        self.d_model = d_model
        self.att_nhead = att_nhead

        d_att = d_model // 2 if att_nhead == 1 else d_model // att_nhead
        self.d_att = d_att
        self.layer_idx = layer_idx

        # Long Short-Term Attention
        self.norm1 = _get_norm(d_model)
        self.linear_QV = nn.Linear(d_model, d_att * att_nhead + expand_d_model)
        self.linear_U = nn.Linear(d_model, expand_d_model)

        if layer_idx == 0:
            self.linear_ID_V = nn.Linear(d_model, expand_d_model)
        else:
            self.id_norm1 = _get_norm(d_model)
            self.linear_ID_V = nn.Linear(d_model * 2, expand_d_model)
            self.linear_ID_U = nn.Linear(d_model, expand_d_model)

        self.long_term_attn = GatedPropagation(d_qk=self.d_model,
                                    d_vu=self.d_model * 2,
                                    num_head=att_nhead,
                                    use_linear=False,
                                    dropout=lt_dropout,
                                    d_att=d_att,
                                    top_k=topk,
                                    expand_ratio=expand_ratio)

        self.short_term_attn = LocalGatedPropagation(d_qk=self.d_model,
                                          d_vu=self.d_model * 2,
                                          num_head=att_nhead,
                                          dilation=local_dilation,
                                          use_linear=False,
                                          dropout=st_dropout,
                                          d_att=d_att,
                                          max_dis=max_local_dis,
                                          expand_ratio=expand_ratio)

        self.lst_dropout = nn.Dropout(max(lt_dropout, st_dropout), True)
        self.droppath_lst = droppath_lst

        # Self-attention
        self.norm2 = _get_norm(d_model)
        self.id_norm2 = _get_norm(d_model)
        self.self_attn = GatedPropagation(d_model * 2,
                               d_model * 2,
                               self_nhead,
                               d_att=d_att)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                tgt_id=None,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30)):

        # Long Short-Term Attention
        _tgt = self.norm1(tgt)

        curr_QV = self.linear_QV(_tgt)
        curr_QV = torch.split(
            curr_QV, [self.d_att * self.att_nhead, self.expand_d_model], dim=2)
        curr_Q = curr_K = curr_QV[0]
        local_Q = seq_to_2d(curr_Q, size_2d)
        curr_V = silu(curr_QV[1])
        curr_U = self.linear_U(_tgt)

        if tgt_id is None:
            tgt_id = 0
            cat_curr_U = torch.cat(
                [silu(curr_U), torch.ones_like(curr_U)], dim=-1)
            curr_ID_V = None
        else:
            _tgt_id = self.id_norm1(tgt_id)
            curr_ID_V = _tgt_id
            curr_ID_U = self.linear_ID_U(_tgt_id)
            cat_curr_U = silu(torch.cat([curr_U, curr_ID_U], dim=-1))

        if curr_id_emb is not None:
            global_K, global_V = curr_K, curr_V
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)

            _, global_ID_V = self.fuse_key_value_id(None, curr_ID_V,
                                                    curr_id_emb)
            local_ID_V = seq_to_2d(global_ID_V, size_2d)
        else:
            global_K, global_V, _, global_ID_V = long_term_memory
            local_K, local_V, _, local_ID_V = short_term_memory

        cat_global_V = torch.cat([global_V, global_ID_V], dim=-1)
        cat_local_V = torch.cat([local_V, local_ID_V], dim=1)

        cat_tgt2, _ = self.long_term_attn(curr_Q, global_K, cat_global_V,
                                          cat_curr_U, size_2d)
        cat_tgt3, _ = self.short_term_attn(local_Q, local_K, cat_local_V,
                                           cat_curr_U, size_2d)

        tgt2, tgt_id2 = torch.split(cat_tgt2, self.d_model, dim=-1)
        tgt3, tgt_id3 = torch.split(cat_tgt3, self.d_model, dim=-1)

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
            tgt_id = tgt_id + self.droppath(tgt_id2 + tgt_id3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3)
            tgt_id = tgt_id + self.lst_dropout(tgt_id2 + tgt_id3)

        # Self-attention
        _tgt = self.norm2(tgt)
        _tgt_id = self.id_norm2(tgt_id)
        q = k = v = u = torch.cat([_tgt, _tgt_id], dim=-1)

        cat_tgt2, _ = self.self_attn(q, k, v, u, size_2d)

        tgt2, tgt_id2 = torch.split(cat_tgt2, self.d_model, dim=-1)

        tgt = tgt + self.droppath(tgt2)
        tgt_id = tgt_id + self.droppath(tgt_id2)

        return tgt, tgt_id, [[curr_K, curr_V, None, curr_ID_V],
                             [global_K, global_V, None, global_ID_V],
                             [local_K, local_V, None, local_ID_V]]

    def fuse_key_value_id(self, key, value, id_emb):
        ID_K = None
        if value is not None:
            ID_V = silu(self.linear_ID_V(torch.cat([value, id_emb], dim=2)))
        else:
            ID_V = silu(self.linear_ID_V(id_emb))
        return ID_K, ID_V

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
