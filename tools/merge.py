from glob import glob
import sys
import numpy as np
import os
from skimage import io
import torch
import torch.nn.functional as F
import tqdm
import argparse

sys.path.append('.')
sys.path.append('..')
from utils.image import _save_mask
from utils.eval import zip_folder

def get_file_list(dir:str) -> list:
    return sorted(glob(dir+'/*'))
def same_len(l:list) -> int:
    if len(l)==0:
        return 0
    same_len = len(l[0])
    for each in l:
        if len(each) != same_len:
            return -1
    return same_len

parser = argparse.ArgumentParser()
parser.add_argument('--ann_dir',type=str,default='../VOS02/datasets/YTB/2018/valid/Annotations')
parser.add_argument('--prob_dir',type=str,default='./result/try2_R50_AOTL/PRE_YTB_DAV/eval/youtubevos2018')
parser.add_argument('--output_dir',type=str,default='./result/try2_R50_AOTL/PRE_YTB_DAV/eval/youtubevos2018/youtubevos2018_val_try2_R50_AOTL_PRE_YTB_DAV_ckpt_unknown_merge/Annotations')
parser.add_argument('--output_prob', action='store_true')
parser.set_defaults(output_prob=False)
parser.add_argument('--method',type=str,default='mean')
parser.add_argument('--use_weight', action='store_true')
parser.add_argument('--davis_480p',type=bool,default=False)
parser.set_defaults(use_weight=False)
args = parser.parse_args()

method = args.method
ann_dir = args.ann_dir
prob_dir = args.prob_dir
use_weight = args.use_weight
dir_list = glob(prob_dir+'/*_prob')
dir_list = [d + '/Annotations' for d in dir_list if 'merge' not in d]
print(len(dir_list),dir_list)
# dir_list = [
#     './result/try2_R50_AOTL/PRE_YTB_DAV/eval/youtubevos2018/youtubevos2018_val_try2_R50_AOTL_PRE_YTB_DAV_ckpt_unknown_prob/Annotations',
#     './result/try2_R50_AOTL/PRE_YTB_DAV/eval/youtubevos2018/youtubevos2018_val_try2_R50_AOTL_PRE_YTB_DAV_ckpt_unknown_flip_prob/Annotations'
#     './result/try_SwinB_AOTv3/PRE_YTB_DAV/eval/youtubevos2018/youtubevos2018_val_try_SwinB_AOTv3_PRE_YTB_DAV_ckpt_1100_ms_1dot2_prob/Annotations',
#     './result/try_SwinB_AOTv3/PRE_YTB_DAV/eval/youtubevos2018/youtubevos2018_val_try_SwinB_AOTv3_PRE_YTB_DAV_ckpt_1100_flip_prob/Annotations'
# ]
output_prob = args.output_prob
output_dir = args.output_dir
if args.davis_480p:
    output_dir_ann = output_dir[:-5]
else:
    output_dir_ann = output_dir[:-12]
if not os.path.exists(output_dir_ann):
    os.makedirs(output_dir_ann)
if not output_prob:
    os.system('cp -r '+ann_dir+' '+output_dir_ann)

dir_num = len(dir_list)
seq_list = []
for dir in dir_list:
    dir_seqs = get_file_list(dir)
    seq_list.append(dir_seqs)

# check seqs num
seq_num = same_len(seq_list)
if seq_num<1:
    raise 'no seq error'
print(seq_num,'seqs')

# add weights for seqs
if use_weight:
    seq_weights = []
    for seq in dir_list:
        if 'vt6' in seq:
            seq_weights.append(0.4)
        elif 'vt8' in seq:
            seq_weights.append(0.3)
        elif 'vt4' in seq:
            seq_weights.append(0.2)
        elif 'vt9' in seq:
            seq_weights.append(0.1)
    if len(seq_weights) != dir_num:
        raise 'invalid weights!'
    else:
        print(seq_weights)

# merge
for i in tqdm.trange(seq_num):
    seq_name = seq_list[0][i].split('/')[-1]
    # print('merging seq',seq_name)
    output_path = os.path.join(output_dir,seq_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # get size
    ann_mask = io.imread(glob(args.ann_dir+'/'+seq_name+'/*')[0])
    mask_size = ann_mask.shape[:2]

    # collect npy dirs
    npy_list = []
    for j in range(dir_num):
        npy_list.append(get_file_list(seq_list[j][i]))
    npy_num = same_len(npy_list)
    if npy_num<1:
        raise 'no npy error'
    
    for idx in range(npy_num):
        img_npys = []
        if output_prob:
            for j in range(dir_num):
                img_npy = np.load(npy_list[j][idx])
                img_npys.append(img_npy)
            mean_npy = np.mean(img_npys,axis=0).astype(np.uint16)
            # save prob
            mean_npy_name = npy_list[j][idx].split('/')[-1]
            mean_npy_path = os.path.join(output_path,mean_npy_name)
            np.save(mean_npy_path,mean_npy)
        else:
            for j in range(dir_num):
                max = np.iinfo(np.uint16).max
                img_npy = np.load(npy_list[j][idx]).astype(np.float32) #(1,n,h,w)

                if img_npy.shape[1:] != mask_size:
                    img_npy /= max
                    img_tensor = torch.Tensor(img_npy)
                    img_tensor = F.interpolate(img_tensor.unsqueeze(0),size=mask_size,mode='nearest')#,align_corners=True)
                    img_tensor = img_tensor.squeeze(0)
                    img_npy = img_tensor.numpy()
                if use_weight:
                    img_npy = img_npy * seq_weights[j]
                img_npys.append(img_npy)
            if method == 'mean':
                mean_npy = np.mean(img_npys,axis=0) #(n,h,w)
                mask = np.argmax(mean_npy,axis=0).astype(np.uint8)
            elif method == 'vote':
                vote_npys = []
                # for npy in img_npys:
                #     indices = np.expand_dims(np.argmax(npy,axis=0),axis=0)
                #     vote_npy = np.zeros_like(npy)
                #     np.put_along_axis(vote_npy,indices,1,axis=0)
                #     vote_npys.append(vote_npy)
                # mean_npy = np.mean(vote_npys,axis=0)
                # mask = np.argmax(mean_npy,axis=0).astype(np.uint8)
                for npy in img_npys:
                    img_max = np.expand_dims(np.max(npy,axis=0),axis=0)
                    img_max_c = np.tile(img_max,(npy.shape[0],1,1))
                    vote_npy = (npy - img_max_c) == 0
                    vote_npys.append(vote_npy)
                mean_npy = np.mean(vote_npys,axis=0)
                mask = np.argmax(mean_npy,axis=0).astype(np.uint8)
            # save mask
            mask_name = npy_list[j][idx].split('/')[-1][:-4]+'.png'
            mask_path = os.path.join(output_path,mask_name)
            _save_mask(mask,mask_path)

if args.davis_480p:
    zip_dir = output_dir_ann[:-12]+'.zip'
else:
    zip_dir = output_dir_ann+'.zip'
zip_folder(output_dir, zip_dir)
print('Saving result to {}.'.format(zip_dir))

