from tkinter import N
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import threading
# from torchvision.ops import masks_to_boxes

_palette = [
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 
    128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128, 191,
    0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0, 191, 0, 128,
    191, 0, 0, 64, 128, 128, 64, 128, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0,
    128, 128, 0, 128, 0, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0,
    64, 0, 128, 191, 0, 128, 64, 128, 128, 191, 128, 128, 0, 64, 0, 128, 64, 0, 0,
    191, 0, 128, 191, 0, 0, 64, 128, 128, 64, 128, 64, 192, 128, 64, 64, 128, 64,
    128, 128, 128, 192, 128, 128, 64, 128, 192, 192, 128, 192, 64, 128, 192, 128,
    128, 64, 192, 192, 64, 64, 192, 64, 128, 192, 128, 192, 192, 128, 64, 192, 128,
    128, 192, 192, 64, 192, 192, 128, 192, 64, 192, 64, 64, 128, 64, 128, 192, 64,
    128, 64, 64, 128, 128, 64, 192, 192, 64, 192, 64, 64, 192, 128, 64, 0, 192, 128,
    0, 0, 128, 0, 128, 128, 128, 192, 128, 128, 0, 128, 192, 192, 128, 192, 0, 128,
    192, 128, 128, 0, 192, 192, 0, 0, 192, 0, 128, 192, 128, 192, 192, 128, 0, 192,
    128, 128, 192, 192, 0, 192, 192, 128, 192, 0, 192, 0, 0, 128, 0, 128, 192, 0, 128,
    0, 0, 128, 128, 0, 192, 192, 0, 192, 0, 0, 192, 128, 0, 0, 192, 64, 0, 0, 64, 0, 64,
    64, 64, 192, 64, 64, 0, 64, 192, 192, 64, 192, 0, 64, 192, 64, 64, 0, 192, 192, 0, 0,
    192, 0, 64, 192, 64, 192, 192, 64, 0, 192, 64, 64, 192, 192, 0, 192, 192, 64, 192, 0,
    192, 0, 0, 64, 0, 64, 192, 0, 64, 0, 0, 64, 64, 0, 192, 192, 0, 192, 0, 0, 192, 64, 0,
    0, 128, 64, 0, 0, 64, 0, 64, 64, 64, 128, 64, 64, 0, 64, 128, 128, 64, 128, 0, 64, 128,
    64, 64, 0, 128, 128, 0, 0, 128, 0, 64, 128, 64, 128, 128, 64, 0, 128, 64, 64, 128, 128,
    0, 128, 128, 64, 128, 0, 128, 0, 0, 64, 0, 64, 128, 0, 64, 0, 0, 64, 64, 0, 128, 128, 0,
    128, 0, 0, 128, 64, 0
]


def label2colormap(label):

    m = label.astype(np.uint8)
    r, c = m.shape
    cmap = np.zeros((r, c, 3), dtype=np.uint8)
    cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3 | (m & 64) >> 1
    cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2 | (m & 128) >> 2
    cmap[:, :, 2] = (m & 4) << 5 | (m & 32) << 1
    return cmap


def one_hot_mask(mask, cls_num, add_bk=True):
    if len(mask.size()) == 3:
        mask = mask.unsqueeze(1)
    indices = torch.arange(0, cls_num + 1 if add_bk else cls_num,
                           device=mask.device).view(1, -1, 1, 1)
    return (mask == indices).float()

def split_stuff_thing_mask(mask,obj_mapping):
    '''
    mask: (B,1,H,W)
    obj_mapping: [B dict]
    '''
    # mask initialized as -1, for dummy one hot
    stuff_mask = torch.zeros_like(mask) - 1
    thing_mask = torch.zeros_like(mask) - 1

    
    for b in range(mask.shape[0]):
        mask_b = mask[b]
        obj_dict = obj_mapping[b]
        for k in obj_dict.keys():
            if obj_dict[k][0] == 0: #stuff
                stuff_mask[b][mask_b==k] = obj_dict[k][1]
            else:
                thing_mask[b][mask_b==k] = obj_dict[k][1]
    return stuff_mask, thing_mask

def masked_image(image, colored_mask, mask, alpha=0.7):
    mask = np.expand_dims(mask > 0, axis=0)
    mask = np.repeat(mask, 3, axis=0)
    show_img = (image * alpha + colored_mask *
                (1 - alpha)) * mask + image * (1 - mask)
    return show_img


def save_image(image, path):
    im = Image.fromarray(np.uint8(image * 255.).transpose((1, 2, 0)))
    im.save(path)


def _save_mask(mask, path, squeeze_idx=None):
    if squeeze_idx is not None:
        unsqueezed_mask = mask * 0
        for idx in range(1, len(squeeze_idx)):
            obj_id = squeeze_idx[idx]
            mask_i = mask == idx
            unsqueezed_mask += (mask_i * obj_id).astype(np.uint8)
        mask = unsqueezed_mask
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(_palette)
    mask.save(path)


def save_mask(mask_tensor, path, squeeze_idx=None):
    mask = mask_tensor.cpu().numpy().astype('uint8')
    threading.Thread(target=_save_mask, args=[mask, path, squeeze_idx]).start()

def save_prob(prob,path,squeeze_idx=None,scale=1):
    if scale != 1:
        # prob = F.interpolate(prob,scale_factor=scale,mode='nearest')
        prob = F.interpolate(prob,scale_factor=scale,mode='bilinear')
    prob = prob.squeeze(0)
    
    if squeeze_idx is None:
        idx_prob = prob
    else:
        c,h,w = prob.shape
        idx_prob = torch.zeros((max(squeeze_idx)+1,h,w),device=prob.device,dtype=prob.dtype)
        for i in range(len(squeeze_idx)):
            idx_prob[squeeze_idx[i]] = prob[i]
        # idx_prob = idx_prob[:max(squeeze_idx)+1]
    max_val = np.iinfo(np.uint16).max
    idx_prob = (idx_prob.cpu().numpy()*max_val).astype(np.uint16)

    np.save(path,idx_prob)
    pass

def save_logit(logit,path,obj_num=None):
    if obj_num is not None:
        logit = logit[:obj_num+1]
    torch.save(logit,path)

def flip_tensor(tensor, dim=0):
    inv_idx = torch.arange(tensor.size(dim) - 1, -1, -1,
                           device=tensor.device).long()
    tensor = tensor.index_select(dim, inv_idx)
    return tensor


def shuffle_obj_mask(mask):

    bs, obj_num, _, _ = mask.size()
    new_masks = []
    for idx in range(bs):
        now_mask = mask[idx]
        random_matrix = torch.eye(obj_num, device=mask.device)
        fg = random_matrix[1:][torch.randperm(obj_num - 1)]
        random_matrix = torch.cat([random_matrix[0:1], fg], dim=0)
        now_mask = torch.einsum('nm,nhw->mhw', random_matrix, now_mask)
        new_masks.append(now_mask)

    return torch.stack(new_masks, dim=0)

def label2box(label):
    '''
    label: tensor[int], shape(1,h,w)
    '''
    box_dict = {}
    
    for i in torch.unique(label):
        if i==0:
            continue
        box_dict[int(i)] = masks_to_boxes(label==i)[0]
    return box_dict

def check_box(box_dict,h,w):
    new_dict = {}
    #check if big obj
    for k in box_dict.keys():
        if (box_dict[k][2]- box_dict[k][0]) > (w * 0.5) or \
            (box_dict[k][3] - box_dict[k][1]) > (h * 0.6):
                continue
        else:
            new_dict[k] = box_dict[k]
    return new_dict

def box_filter(box_dict,label,prob=None,h_ratio=0.5,w_ratio=0.5):
    
    
    h,w = label.shape[2],label.shape[3]
    for k in box_dict.keys():
        obj_filter = torch.zeros_like(label,dtype=torch.uint8)
        center_x = (box_dict[k][0] + box_dict[k][2]) /2
        center_y = (box_dict[k][1] + box_dict[k][3]) /2
        box = [center_x - w*w_ratio/2,center_y - h*h_ratio/2,
                center_x + w*w_ratio/2,center_y + h*h_ratio/2]
        box = list(map(lambda x: int(x), box))
        box[0] = max(box[0],0)
        box[1] = max(box[1],0)
        box[2] = min(box[2],w)
        box[3] = min(box[3],h)
        obj_filter[:,:,box[1]:box[3],box[0]:box[2]] = 1
    
        filter_mask = (label.to(torch.uint8)==k) & (obj_filter==0)
        label[filter_mask] = 0.
        
        if prob != None:
            c = prob.shape[1]
            bk = torch.zeros_like(prob)
            bk[:,0] = 1
            filter_mask_c = filter_mask.repeat(1,c,1,1)
            prob[filter_mask_c] = bk[filter_mask_c]
    
    return [label,prob]

def custom_collate(batch):
    from torch.utils.data._utils.collate import default_collate # for torch < v1.13
    elem = batch[0]
    if isinstance(elem, dict):
        batched_sample = {}
        for k in elem.keys():
            if k != 'meta':
                batched_sample[k] = default_collate([b[k] for b in batch])
            else:
                batched_meta = {}
                for km in elem['meta'].keys():
                    batched_meta[km] = [b['meta'][km] for b in batch]
                batched_sample['meta'] = batched_meta
        return batched_sample
    else:
        return torch.utils.data.default_collate(batch)