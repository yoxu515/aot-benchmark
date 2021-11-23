import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import DropOutLogit


# Long-term attention
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_head=8, dropout=0., use_linear=True):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head

        self.hidden_dim = d_model // num_head
        self.T = (d_model / num_head)**0.5
        self.use_linear = use_linear

        if use_linear:
            self.linear_Q = nn.Linear(d_model, d_model)
            self.linear_K = nn.Linear(d_model, d_model)
            self.linear_V = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.drop_prob = dropout
        self.projection = nn.Linear(d_model, d_model)
        self._init_weight()

    def forward(self, Q, K, V):
        """
        :param Q: A 3d tensor with shape of [T_q, bs, C_q]
        :param K: A 3d tensor with shape of [T_k, bs, C_k]
        :param V: A 3d tensor with shape of [T_v, bs, C_v]
        """
        num_head = self.num_head
        hidden_dim = self.hidden_dim

        bs = Q.size()[1]

        # Linear projections
        if self.use_linear:
            Q = self.linear_Q(Q)
            K = self.linear_K(K)
            V = self.linear_V(V)

        # Scale
        Q = Q / self.T

        # Multi-head
        Q = Q.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
        K = K.view(-1, bs, num_head, hidden_dim).permute(1, 2, 3, 0)
        V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)

        # Multiplication
        QK = Q @ K

        # Activation
        attn = torch.softmax(QK, dim=-1)

        # Dropouts
        attn = self.dropout(attn)

        # Weighted sum
        outputs = (attn @ V).permute(2, 0, 1, 3)

        # Restore shape
        outputs = outputs.reshape(-1, bs, self.d_model)

        outputs = self.projection(outputs)

        return outputs, attn

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


# Short-term attention
class MultiheadLocalAttentionV1(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 dropout=0.,
                 max_dis=7,
                 dilation=1,
                 use_linear=True,
                 enable_corr=True):
        super().__init__()
        self.dilation = dilation
        self.window_size = 2 * max_dis + 1
        self.max_dis = max_dis
        self.num_head = num_head
        self.T = ((d_model / num_head)**0.5)

        self.use_linear = use_linear
        if use_linear:
            self.linear_Q = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_K = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_V = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.relative_emb_k = nn.Conv2d(d_model,
                                        num_head * self.window_size *
                                        self.window_size,
                                        kernel_size=1,
                                        groups=num_head)
        self.relative_emb_v = nn.Parameter(
            torch.zeros([
                self.num_head, d_model // self.num_head,
                self.window_size * self.window_size
            ]))

        self.enable_corr = enable_corr

        if enable_corr:
            from spatial_correlation_sampler import SpatialCorrelationSampler
            self.correlation_sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.window_size,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=self.dilation)

        self.projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.drop_prob = dropout

    def forward(self, q, k, v):
        n, c, h, w = v.size()

        if self.use_linear:
            q = self.linear_Q(q)
            k = self.linear_K(k)
            v = self.linear_V(v)

        hidden_dim = c // self.num_head

        relative_emb = self.relative_emb_k(q)
        memory_mask = torch.ones((1, 1, h, w), device=v.device).float()

        # Scale
        q = q / self.T

        q = q.view(-1, hidden_dim, h, w)
        k = k.reshape(-1, hidden_dim, h, w).contiguous()
        unfolded_v = self.pad_and_unfold(v).view(
            n, self.num_head, hidden_dim, self.window_size * self.window_size,
            h * w) + self.relative_emb_v.unsqueeze(0).unsqueeze(-1)

        relative_emb = relative_emb.view(n, self.num_head,
                                         self.window_size * self.window_size,
                                         h * w)
        unfolded_k_mask = self.pad_and_unfold(memory_mask).bool().view(
            1, 1, self.window_size * self.window_size,
            h * w).expand(n, self.num_head, -1, -1)

        if self.enable_corr:
            qk = self.correlation_sampler(q, k).view(
                n, self.num_head, self.window_size * self.window_size,
                h * w) + relative_emb
        else:
            unfolded_k = self.pad_and_unfold(k).view(
                n * self.num_head, hidden_dim,
                self.window_size * self.window_size, h, w)
            qk = (q.unsqueeze(2) * unfolded_k).sum(dim=1).view(
                n, self.num_head, self.window_size * self.window_size,
                h * w) + relative_emb

        qk_mask = 1 - unfolded_k_mask

        qk -= qk_mask * 1e+8 if qk.dtype == torch.float32 else qk_mask * 1e+4

        local_attn = torch.softmax(qk, dim=2)

        local_attn = self.dropout(local_attn)

        output = (local_attn.unsqueeze(2) * unfolded_v).sum(dim=3).permute(
            3, 0, 1, 2).view(h * w, n, c)

        output = self.projection(output)

        return output, local_attn

    def pad_and_unfold(self, x):
        pad_pixel = self.max_dis * self.dilation
        x = F.pad(x, (pad_pixel, pad_pixel, pad_pixel, pad_pixel),
                  mode='constant',
                  value=0)
        x = F.unfold(x,
                     kernel_size=(self.window_size, self.window_size),
                     stride=(1, 1),
                     dilation=self.dilation)
        return x


class MultiheadLocalAttentionV2(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 dropout=0.,
                 max_dis=7,
                 dilation=1,
                 use_linear=True,
                 enable_corr=True):
        super().__init__()
        self.dilation = dilation
        self.window_size = 2 * max_dis + 1
        self.max_dis = max_dis
        self.num_head = num_head
        self.T = (d_model / num_head)**0.5

        self.use_linear = use_linear
        if use_linear:
            self.linear_Q = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_K = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_V = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.relative_emb_k = nn.Conv2d(d_model,
                                        num_head * self.window_size *
                                        self.window_size,
                                        kernel_size=1,
                                        groups=num_head)
        self.relative_emb_v = nn.Parameter(
            torch.zeros([
                self.num_head, d_model // self.num_head,
                self.window_size * self.window_size
            ]))

        self.enable_corr = enable_corr

        if enable_corr:
            from spatial_correlation_sampler import SpatialCorrelationSampler
            self.correlation_sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.window_size,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=self.dilation)

        self.projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.drop_prob = dropout

        self.local_mask = None
        self.last_size_2d = None
        self.qk_mask = None

    def forward(self, q, k, v):
        n, c, h, w = v.size()

        if self.use_linear:
            q = self.linear_Q(q)
            k = self.linear_K(k)
            v = self.linear_V(v)

        hidden_dim = c // self.num_head

        if self.qk_mask is not None and (h, w) == self.last_size_2d:
            qk_mask = self.qk_mask
        else:
            memory_mask = torch.ones((1, 1, h, w), device=v.device).float()
            unfolded_k_mask = self.pad_and_unfold(memory_mask).view(
                1, 1, self.window_size * self.window_size, h * w)
            qk_mask = 1 - unfolded_k_mask
            self.qk_mask = qk_mask

        relative_emb = self.relative_emb_k(q)

        # Scale
        q = q / self.T

        q = q.view(-1, hidden_dim, h, w)
        k = k.view(-1, hidden_dim, h, w)
        v = v.view(-1, self.num_head, hidden_dim, h * w)

        relative_emb = relative_emb.view(n, self.num_head,
                                         self.window_size * self.window_size,
                                         h * w)

        if self.enable_corr:
            qk = self.correlation_sampler(q, k).view(
                n, self.num_head, self.window_size * self.window_size,
                h * w) + relative_emb
        else:
            unfolded_k = self.pad_and_unfold(k).view(
                n * self.num_head, hidden_dim,
                self.window_size * self.window_size, h, w)
            qk = (q.unsqueeze(2) * unfolded_k).sum(dim=1).view(
                n, self.num_head, self.window_size * self.window_size,
                h * w) + relative_emb

        qk -= qk_mask * 1e+8 if qk.dtype == torch.float32 else qk_mask * 1e+4

        local_attn = torch.softmax(qk, dim=2)

        local_attn = self.dropout(local_attn)

        agg_bias = torch.einsum('bhwn,hcw->bhnc', local_attn,
                                self.relative_emb_v)

        global_attn = self.local2global(local_attn, h, w)

        agg_value = (global_attn @ v.transpose(-2, -1))

        output = (agg_value + agg_bias).permute(2, 0, 1,
                                                3).reshape(h * w, n, c)

        output = self.projection(output)

        self.last_size_2d = (h, w)
        return output, local_attn

    def local2global(self, local_attn, height, width):
        batch_size = local_attn.size()[0]

        pad_height = height + 2 * self.max_dis
        pad_width = width + 2 * self.max_dis

        if self.local_mask is not None and (height,
                                            width) == self.last_size_2d:
            local_mask = self.local_mask
        else:
            ky, kx = torch.meshgrid([
                torch.arange(0, pad_height, device=local_attn.device),
                torch.arange(0, pad_width, device=local_attn.device)
            ])
            qy, qx = torch.meshgrid([
                torch.arange(0, height, device=local_attn.device),
                torch.arange(0, width, device=local_attn.device)
            ])

            offset_y = qy.reshape(-1, 1) - ky.reshape(1, -1) + self.max_dis
            offset_x = qx.reshape(-1, 1) - kx.reshape(1, -1) + self.max_dis

            local_mask = (offset_y.abs() <= self.max_dis) & (offset_x.abs() <=
                                                             self.max_dis)
            local_mask = local_mask.view(1, 1, height * width, pad_height,
                                         pad_width)
            self.local_mask = local_mask

        global_attn = torch.zeros(
            (batch_size, self.num_head, height * width, pad_height, pad_width),
            device=local_attn.device)
        global_attn[local_mask.expand(batch_size, self.num_head,
                                      -1, -1, -1)] = local_attn.transpose(
                                          -1, -2).reshape(-1)
        global_attn = global_attn[:, :, :, self.max_dis:-self.max_dis,
                                  self.max_dis:-self.max_dis].reshape(
                                      batch_size, self.num_head,
                                      height * width, height * width)

        return global_attn

    def pad_and_unfold(self, x):
        pad_pixel = self.max_dis * self.dilation
        x = F.pad(x, (pad_pixel, pad_pixel, pad_pixel, pad_pixel),
                  mode='constant',
                  value=0)
        x = F.unfold(x,
                     kernel_size=(self.window_size, self.window_size),
                     stride=(1, 1),
                     dilation=self.dilation)
        return x


class MultiheadLocalAttentionV3(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 dropout=0.,
                 max_dis=7,
                 dilation=1,
                 use_linear=True):
        super().__init__()
        self.dilation = dilation
        self.window_size = 2 * max_dis + 1
        self.max_dis = max_dis
        self.num_head = num_head
        self.T = ((d_model / num_head)**0.5)

        self.use_linear = use_linear
        if use_linear:
            self.linear_Q = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_K = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_V = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.relative_emb_k = nn.Conv2d(d_model,
                                        num_head * self.window_size *
                                        self.window_size,
                                        kernel_size=1,
                                        groups=num_head)
        self.relative_emb_v = nn.Parameter(
            torch.zeros([
                self.num_head, d_model // self.num_head,
                self.window_size * self.window_size
            ]))

        self.projection = nn.Linear(d_model, d_model)
        self.dropout = DropOutLogit(dropout)

        self.padded_local_mask = None
        self.local_mask = None
        self.last_size_2d = None
        self.qk_mask = None

    def forward(self, q, k, v):
        n, c, h, w = q.size()

        if self.use_linear:
            q = self.linear_Q(q)
            k = self.linear_K(k)
            v = self.linear_V(v)

        hidden_dim = c // self.num_head

        relative_emb = self.relative_emb_k(q)
        relative_emb = relative_emb.view(n, self.num_head,
                                         self.window_size * self.window_size,
                                         h * w)
        padded_local_mask, local_mask = self.compute_mask(h,
                                                          w,
                                                          device=q.device)
        qk_mask = (~padded_local_mask).float()

        # Scale
        q = q / self.T

        q = q.view(-1, self.num_head, hidden_dim, h * w)
        k = k.view(-1, self.num_head, hidden_dim, h * w)
        v = v.view(-1, self.num_head, hidden_dim, h * w)

        qk = q.transpose(-1, -2) @ k  # [B, nH, kL, qL]

        pad_pixel = self.max_dis * self.dilation

        padded_qk = F.pad(qk.view(-1, self.num_head, h * w, h, w),
                          (pad_pixel, pad_pixel, pad_pixel, pad_pixel),
                          mode='constant',
                          value=-1e+8 if qk.dtype == torch.float32 else -1e+4)

        qk_mask = qk_mask * 1e+8 if (padded_qk.dtype
                                     == torch.float32) else qk_mask * 1e+4
        padded_qk = padded_qk - qk_mask

        padded_qk[padded_local_mask.expand(n, self.num_head, -1, -1,
                                           -1)] += relative_emb.transpose(
                                               -1, -2).reshape(-1)
        padded_qk = self.dropout(padded_qk)

        local_qk = padded_qk[padded_local_mask.expand(n, self.num_head, -1, -1,
                                                      -1)]

        global_qk = padded_qk[:, :, :, self.max_dis:-self.max_dis,
                              self.max_dis:-self.max_dis].reshape(
                                  n, self.num_head, h * w, h * w)

        local_attn = torch.softmax(local_qk.reshape(
            n, self.num_head, h * w, self.window_size * self.window_size),
                                   dim=3)
        global_attn = torch.softmax(global_qk, dim=3)

        agg_bias = torch.einsum('bhnw,hcw->bhnc', local_attn,
                                self.relative_emb_v)

        agg_value = (global_attn @ v.transpose(-2, -1))

        output = (agg_value + agg_bias).permute(2, 0, 1,
                                                3).reshape(h * w, n, c)

        output = self.projection(output)

        self.last_size_2d = (h, w)
        return output, local_attn

    def compute_mask(self, height, width, device=None):
        pad_height = height + 2 * self.max_dis
        pad_width = width + 2 * self.max_dis

        if self.padded_local_mask is not None and (height,
                                                   width) == self.last_size_2d:
            padded_local_mask = self.padded_local_mask
            local_mask = self.local_mask

        else:
            ky, kx = torch.meshgrid([
                torch.arange(0, pad_height, device=device),
                torch.arange(0, pad_width, device=device)
            ])
            qy, qx = torch.meshgrid([
                torch.arange(0, height, device=device),
                torch.arange(0, width, device=device)
            ])

            qy = qy.reshape(-1, 1)
            qx = qx.reshape(-1, 1)
            offset_y = qy - ky.reshape(1, -1) + self.max_dis
            offset_x = qx - kx.reshape(1, -1) + self.max_dis
            padded_local_mask = (offset_y.abs() <= self.max_dis) & (
                offset_x.abs() <= self.max_dis)
            padded_local_mask = padded_local_mask.view(1, 1, height * width,
                                                       pad_height, pad_width)
            local_mask = padded_local_mask[:, :, :, self.max_dis:-self.max_dis,
                                           self.max_dis:-self.max_dis]
            pad_pixel = self.max_dis * self.dilation
            local_mask = F.pad(local_mask.float(),
                               (pad_pixel, pad_pixel, pad_pixel, pad_pixel),
                               mode='constant',
                               value=0).view(1, 1, height * width, pad_height,
                                             pad_width)
            self.padded_local_mask = padded_local_mask
            self.local_mask = local_mask

        return padded_local_mask, local_mask
