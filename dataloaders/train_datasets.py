from __future__ import division
import os
from glob import glob
import json
import random
import cv2
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as TF
from dataloaders import tps

import dataloaders.image_transforms as IT

cv2.setNumThreads(0)


def _get_images(sample):
    return [sample['ref_img'], sample['prev_img']] + sample['curr_img']


def _get_labels(sample):
    return [sample['ref_label'], sample['prev_label']] + sample['curr_label']


def _merge_sample(sample1, sample2, min_obj_pixels=100, max_obj_n=10):

    sample1_images = _get_images(sample1)
    sample2_images = _get_images(sample2)

    sample1_labels = _get_labels(sample1)
    sample2_labels = _get_labels(sample2)

    obj_idx = torch.arange(0, max_obj_n * 2 + 1).view(max_obj_n * 2 + 1, 1, 1)
    selected_idx = None
    selected_obj = None

    all_img = []
    all_mask = []
    for idx, (s1_img, s2_img, s1_label, s2_label) in enumerate(
            zip(sample1_images, sample2_images, sample1_labels,
                sample2_labels)):
        s2_fg = (s2_label > 0).float()
        s2_bg = 1 - s2_fg
        merged_img = s1_img * s2_bg + s2_img * s2_fg
        merged_mask = s1_label * s2_bg.long() + (
            (s2_label + max_obj_n) * s2_fg.long())
        merged_mask = (merged_mask == obj_idx).float()
        if idx == 0:
            after_merge_pixels = merged_mask.sum(dim=(1, 2), keepdim=True)
            selected_idx = after_merge_pixels > min_obj_pixels
            selected_idx[0] = True
            obj_num = selected_idx.sum().int().item() - 1
            selected_idx = selected_idx.expand(-1,
                                               s1_label.size()[1],
                                               s1_label.size()[2])
            if obj_num > max_obj_n:
                selected_obj = list(range(1, obj_num + 1))
                random.shuffle(selected_obj)
                selected_obj = [0] + selected_obj[:max_obj_n]

        merged_mask = merged_mask[selected_idx].view(obj_num + 1,
                                                     s1_label.size()[1],
                                                     s1_label.size()[2])
        if obj_num > max_obj_n:
            merged_mask = merged_mask[selected_obj]
        merged_mask[0] += 0.1
        merged_mask = torch.argmax(merged_mask, dim=0, keepdim=True).long()

        all_img.append(merged_img)
        all_mask.append(merged_mask)

    sample = {
        'ref_img': all_img[0],
        'prev_img': all_img[1],
        'curr_img': all_img[2:],
        'ref_label': all_mask[0],
        'prev_label': all_mask[1],
        'curr_label': all_mask[2:]
    }
    sample['meta'] = sample1['meta']
    sample['meta']['obj_num'] = min(obj_num, max_obj_n)
    return sample


class StaticTrain(Dataset):
    def __init__(self,
                 root,
                 output_size,
                 seq_len=5,
                 max_obj_n=10,
                 dynamic_merge=True,
                 merge_prob=1.0,
                 strong_aug=False,
                 tps_prob=0.0,
                 tps_scale=0.02):
        self.root = root
        self.clip_n = seq_len
        self.output_size = output_size
        self.max_obj_n = max_obj_n

        self.dynamic_merge = dynamic_merge
        self.merge_prob = merge_prob

        self.img_list = list()
        self.mask_list = list()

        dataset_list = list()
        lines = ['COCO', 'ECSSD', 'MSRA10K', 'PASCAL-S', 'PASCALVOC2012']
        for line in lines:
            dataset_name = line.strip()

            img_dir = os.path.join(root, 'JPEGImages', dataset_name)
            mask_dir = os.path.join(root, 'Annotations', dataset_name)

            img_list = sorted(glob(os.path.join(img_dir, '*.jpg'))) + \
                sorted(glob(os.path.join(img_dir, '*.png')))
            mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

            if len(img_list) > 0:
                if len(img_list) == len(mask_list):
                    dataset_list.append(dataset_name)
                    self.img_list += img_list
                    self.mask_list += mask_list
                    print(f'\t{dataset_name}: {len(img_list)} imgs.')
                else:
                    print(
                        f'\tPreTrain dataset {dataset_name} has {len(img_list)} imgs and {len(mask_list)} annots. Not match! Skip.'
                    )
            else:
                print(
                    f'\tPreTrain dataset {dataset_name} doesn\'t exist. Skip.')

        print(
            f'{len(self.img_list)} imgs are used for PreTrain. They are from {dataset_list}.'
        )

        self.pre_random_horizontal_flip = IT.RandomHorizontalFlip(0.5)

        self.random_horizontal_flip = IT.RandomHorizontalFlip(0.3)
        if not strong_aug:
            self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.03)
            self.gray_scale = None
            self.blur = None
        else:
            self.color_jitter = TF.RandomApply(
                [TF.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)
            self.gray_scale = TF.RandomGrayscale(p=0.2)
            self.blur = TF.RandomApply([IT.GaussianBlur([.1, 2.])], p=0.3)
            self.random_affine = IT.RandomAffine(degrees=20,
                                                translate=(0.1, 0.1),
                                                scale=(0.9, 1.1),
                                                shear=10,
                                                resample=Image.BICUBIC,
                                                fillcolor=(124, 116, 104))
        self.random_affine = IT.RandomAffine(degrees=20,
                                             translate=(0.1, 0.1),
                                             scale=(0.9, 1.1),
                                             shear=10,
                                             resample=Image.BICUBIC,
                                             fillcolor=(124, 116, 104))
        self.tps_prob = tps_prob
        self.tps_scale = tps_scale
        base_ratio = float(output_size[1]) / output_size[0]
        self.random_resize_crop = IT.RandomResizedCrop(
            output_size, (0.8, 1),
            ratio=(base_ratio * 3. / 4., base_ratio * 4. / 3.),
            interpolation=Image.BICUBIC)
        self.to_tensor = TF.ToTensor()
        self.to_onehot = IT.ToOnehot(max_obj_n, shuffle=True)
        self.normalize = TF.Normalize((0.485, 0.456, 0.406),
                                      (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.img_list)

    def load_image_in_PIL(self, path, mode='RGB'):
        img = Image.open(path)
        img.load()  # Very important for loading large image
        return img.convert(mode)

    def sample_sequence(self, idx):
        img_pil = self.load_image_in_PIL(self.img_list[idx], 'RGB')
        mask_pil = self.load_image_in_PIL(self.mask_list[idx], 'P')

        frames = []
        masks = []

        img_pil, mask_pil = self.pre_random_horizontal_flip(img_pil, mask_pil)

        for i in range(self.clip_n):
            img, mask = img_pil, mask_pil

            if i > 0:
                img, mask = self.random_horizontal_flip(img, mask)
                img = self.color_jitter(img)
                img, mask = self.random_affine(img, mask)

            img, mask = self.random_resize_crop(img, mask)

            if self.gray_scale is not None:
                img = self.gray_scale(img)
            if self.blur is not None:
                img = self.blur(img)
            if np.random.rand() < self.tps_prob:
                img,mask = tps.random_tps_warp(img, mask, scale=self.tps_scale)
                # print('tps!')

            mask = np.array(mask, np.uint8)

            if i == 0:
                mask, obj_list = self.to_onehot(mask)
                obj_num = len(obj_list)
            else:
                mask, _ = self.to_onehot(mask, obj_list)

            mask = torch.argmax(mask, dim=0, keepdim=True)

            frames.append(self.normalize(self.to_tensor(img)))
            masks.append(mask)

        sample = {
            'ref_img': frames[0],
            'prev_img': frames[1],
            'curr_img': frames[2:],
            'ref_label': masks[0],
            'prev_label': masks[1],
            'curr_label': masks[2:]
        }
        sample['meta'] = {
            'seq_name': self.img_list[idx],
            'frame_num': 1,
            'obj_num': obj_num
        }

        return sample

    def __getitem__(self, idx):
        sample1 = self.sample_sequence(idx)

        if self.dynamic_merge and (sample1['meta']['obj_num'] == 0
                                   or random.random() < self.merge_prob):
            rand_idx = np.random.randint(len(self.img_list))
            while (rand_idx == idx):
                rand_idx = np.random.randint(len(self.img_list))

            sample2 = self.sample_sequence(rand_idx)

            sample = self.merge_sample(sample1, sample2)
        else:
            sample = sample1

        return sample

    def merge_sample(self, sample1, sample2, min_obj_pixels=100):
        return _merge_sample(sample1, sample2, min_obj_pixels, self.max_obj_n)


class VOSTrain(Dataset):
    def __init__(self,
                 image_root,
                 label_root,
                 imglistdic,
                 transform=None,
                 rgb=True,
                 repeat_time=1,
                 rand_gap=3,
                 seq_len=5,
                 rand_reverse=True,
                 dynamic_merge=True,
                 enable_prev_frame=False,
                 merge_prob=0.3,
                 max_obj_n=10,
                 balance_sample=False,
                 balance_ratio=0.0,
                 freq_dict=None):
        self.image_root = image_root
        self.label_root = label_root
        self.rand_gap = rand_gap
        self.seq_len = seq_len
        self.rand_reverse = rand_reverse
        self.repeat_time = repeat_time
        self.transform = transform
        self.dynamic_merge = dynamic_merge
        self.merge_prob = merge_prob
        self.enable_prev_frame = enable_prev_frame
        self.max_obj_n = max_obj_n
        self.rgb = rgb
        self.imglistdic = imglistdic
        self.seqs = list(self.imglistdic.keys())
        self.balance_sample = balance_sample
        self.balance_ratio = balance_ratio
        self.freq_dict = freq_dict
        if self.balance_sample:
            p = self.balance_ratio
            freq_dict = {}
            if self.freq_dict is None:
                raise ValueError
            for k in self.freq_dict.keys():
                freq_dict[k] = self.freq_dict[k]**p
                
            self.probs = []
            for seq in self.seqs:
                self.probs.append(freq_dict[seq])
        
        print('Video Num: {} X {}'.format(len(self.seqs), self.repeat_time))

    def __len__(self):
        return int(len(self.seqs) * self.repeat_time)

    def reverse_seq(self, imagelist, lablist):
        if np.random.randint(2) == 1:
            imagelist = imagelist[::-1]
            lablist = lablist[::-1]
        return imagelist, lablist

    def get_ref_index(self,
                      seqname,
                      lablist,
                      objs,
                      min_fg_pixels=200,
                      max_try=5):
        bad_indices = []
        for _ in range(max_try):
            ref_index = np.random.randint(len(lablist))
            if ref_index in bad_indices:
                continue
            ref_label = Image.open(
                os.path.join(self.label_root, seqname, lablist[ref_index]))
            ref_label = np.array(ref_label, dtype=np.uint8)
            ref_objs = list(np.unique(ref_label))
            is_consistent = True
            for obj in ref_objs:
                if obj == 0:
                    continue
                if obj not in objs:
                    is_consistent = False
            xs, ys = np.nonzero(ref_label)
            if len(xs) > min_fg_pixels and is_consistent:
                break
            bad_indices.append(ref_index)
        return ref_index

    def get_ref_index_v2(self,
                         seqname,
                         lablist,
                         min_fg_pixels=200,
                         max_try=20,
                         total_gap=0):
        search_range = len(lablist) - total_gap
        if search_range <= 1:
            return 0
        bad_indices = []
        for _ in range(max_try):
            ref_index = np.random.randint(search_range)
            if ref_index in bad_indices:
                continue
            ref_label = Image.open(
                os.path.join(self.label_root, seqname, lablist[ref_index]))
            ref_label = np.array(ref_label, dtype=np.uint8)
            xs, ys = np.nonzero(ref_label)
            if len(xs) > min_fg_pixels:
                break
            bad_indices.append(ref_index)
        return ref_index

    def get_curr_gaps(self, seq_len, max_gap=999, max_try=10):
        for _ in range(max_try):
            curr_gaps = []
            total_gap = 0
            for _ in range(seq_len):
                gap = int(np.random.randint(self.rand_gap) + 1)
                total_gap += gap
                curr_gaps.append(gap)
            if total_gap <= max_gap:
                break
        return curr_gaps, total_gap

    def get_prev_index(self, lablist, total_gap):
        search_range = len(lablist) - total_gap
        if search_range > 1:
            prev_index = np.random.randint(search_range)
        else:
            prev_index = 0
        return prev_index

    def check_index(self, total_len, index, allow_reflect=True):
        if total_len <= 1:
            return 0

        if index < 0:
            if allow_reflect:
                index = -index
                index = self.check_index(total_len, index, True)
            else:
                index = 0
        elif index >= total_len:
            if allow_reflect:
                index = 2 * (total_len - 1) - index
                index = self.check_index(total_len, index, True)
            else:
                index = total_len - 1

        return index

    def get_curr_indices(self, lablist, prev_index, gaps):
        total_len = len(lablist)
        curr_indices = []
        now_index = prev_index
        for gap in gaps:
            now_index += gap
            curr_indices.append(self.check_index(total_len, now_index))
        return curr_indices

    def get_image_label(self, seqname, imagelist, lablist, index):
        image = cv2.imread(
            os.path.join(self.image_root, seqname, imagelist[index]))
        image = np.array(image, dtype=np.float32)

        if self.rgb:
            image = image[:, :, [2, 1, 0]]

        label = Image.open(
            os.path.join(self.label_root, seqname, lablist[index]))
        label = np.array(label, dtype=np.uint8)

        return image, label

    def sample_sequence(self, idx):
        idx = idx % len(self.seqs)
        seqname = self.seqs[idx]
        imagelist, lablist = self.imglistdic[seqname]
        frame_num = len(imagelist)
        if self.rand_reverse:
            imagelist, lablist = self.reverse_seq(imagelist, lablist)

        is_consistent = False
        max_try = 5
        try_step = 0
        while (is_consistent is False and try_step < max_try):
            try_step += 1

            # generate random gaps
            curr_gaps, total_gap = self.get_curr_gaps(self.seq_len - 1)

            if self.enable_prev_frame:  # prev frame is randomly sampled
                # get prev frame
                prev_index = self.get_prev_index(lablist, total_gap)
                prev_image, prev_label = self.get_image_label(
                    seqname, imagelist, lablist, prev_index)
                prev_objs = list(np.unique(prev_label))

                # get curr frames
                curr_indices = self.get_curr_indices(lablist, prev_index,
                                                     curr_gaps)
                curr_images, curr_labels, curr_objs = [], [], []
                for curr_index in curr_indices:
                    curr_image, curr_label = self.get_image_label(
                        seqname, imagelist, lablist, curr_index)
                    c_objs = list(np.unique(curr_label))
                    curr_images.append(curr_image)
                    curr_labels.append(curr_label)
                    curr_objs.extend(c_objs)

                objs = list(np.unique(prev_objs + curr_objs))

                start_index = prev_index
                end_index = max(curr_indices)
                # get ref frame
                _try_step = 0
                ref_index = self.get_ref_index_v2(seqname, lablist)
                while (ref_index > start_index and ref_index <= end_index
                       and _try_step < max_try):
                    _try_step += 1
                    ref_index = self.get_ref_index_v2(seqname, lablist)
                ref_image, ref_label = self.get_image_label(
                    seqname, imagelist, lablist, ref_index)
                ref_objs = list(np.unique(ref_label))
            else:  # prev frame is next to ref frame
                # get ref frame
                ref_index = self.get_ref_index_v2(seqname, lablist)

                ref_image, ref_label = self.get_image_label(
                    seqname, imagelist, lablist, ref_index)
                ref_objs = list(np.unique(ref_label))

                # get curr frames
                curr_indices = self.get_curr_indices(lablist, ref_index,
                                                     curr_gaps)
                curr_images, curr_labels, curr_objs = [], [], []
                for curr_index in curr_indices:
                    curr_image, curr_label = self.get_image_label(
                        seqname, imagelist, lablist, curr_index)
                    c_objs = list(np.unique(curr_label))
                    curr_images.append(curr_image)
                    curr_labels.append(curr_label)
                    curr_objs.extend(c_objs)

                objs = list(np.unique(curr_objs))
                prev_image, prev_label = curr_images[0], curr_labels[0]
                curr_images, curr_labels = curr_images[1:], curr_labels[1:]

            is_consistent = True
            for obj in objs:
                if obj == 0:
                    continue
                if obj not in ref_objs:
                    is_consistent = False
                    break

        # get meta info
        obj_num = list(np.sort(ref_objs))[-1]

        sample = {
            'ref_img': ref_image,
            'prev_img': prev_image,
            'curr_img': curr_images,
            'ref_label': ref_label,
            'prev_label': prev_label,
            'curr_label': curr_labels
        }
        sample['meta'] = {
            'seq_name': seqname,
            'frame_num': frame_num,
            'obj_num': obj_num
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, old_idx):
        if self.balance_sample:
            idx = self.get_balance_idx()
            # print(old_idx,'after balanced',idx)
        else:
            idx = old_idx
        sample1 = self.sample_sequence(idx)
        
        if self.dynamic_merge and (sample1['meta']['obj_num'] == 0
                                   or random.random() < self.merge_prob):
                                   
            if self.balance_sample:
                rand_idx = self.get_balance_idx()
            else:
                rand_idx = np.random.randint(len(self.seqs))
            while (rand_idx == (idx % len(self.seqs))):
                if self.balance_sample:
                    # add an random number to prevent loop mapping from idx to rand_idx
                    rand_idx = self.get_balance_idx()
                else:
                    rand_idx = np.random.randint(len(self.seqs))
            # print(idx, 'random merge with', rand_idx)
            sample2 = self.sample_sequence(rand_idx)

            sample = self.merge_sample(sample1, sample2)
        else:
            sample = sample1

        return sample

    def merge_sample(self, sample1, sample2, min_obj_pixels=100):
        return _merge_sample(sample1, sample2, min_obj_pixels, self.max_obj_n)
    
    def get_balance_idx(self):
        '''
        generate index of a seq acd to freq_dict**p
        p is balance ratio
        '''
        idx = random.choices(range(len(self.seqs)),self.probs)
        return idx[0]


class DAVIS2017_Train(VOSTrain):
    def __init__(self,
                 split=['train'],
                 root='./DAVIS',
                 transform=None,
                 rgb=True,
                 repeat_time=1,
                 full_resolution=True,
                 year=2017,
                 rand_gap=3,
                 seq_len=5,
                 rand_reverse=True,
                 dynamic_merge=True,
                 enable_prev_frame=False,
                 max_obj_n=10,
                 merge_prob=0.3):
        if full_resolution:
            resolution = 'Full-Resolution'
            if not os.path.exists(os.path.join(root, 'JPEGImages',
                                               resolution)):
                print('No Full-Resolution, use 480p instead.')
                resolution = '480p'
        else:
            resolution = '480p'
        image_root = os.path.join(root, 'JPEGImages', resolution)
        label_root = os.path.join(root, 'Annotations', resolution)
        seq_names = []
        for spt in split:
            with open(os.path.join(root, 'ImageSets', str(year),
                                   spt + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            seq_names.extend(seqs_tmp)
        imglistdic = {}
        for seq_name in seq_names:
            images = list(
                np.sort(os.listdir(os.path.join(image_root, seq_name))))
            labels = list(
                np.sort(os.listdir(os.path.join(label_root, seq_name))))
            imglistdic[seq_name] = (images, labels)

        super(DAVIS2017_Train, self).__init__(image_root,
                                              label_root,
                                              imglistdic,
                                              transform,
                                              rgb,
                                              repeat_time,
                                              rand_gap,
                                              seq_len,
                                              rand_reverse,
                                              dynamic_merge,
                                              enable_prev_frame,
                                              merge_prob=merge_prob,
                                              max_obj_n=max_obj_n)


class YOUTUBEVOS_Train(VOSTrain):
    def __init__(self,
                 root='./datasets/YTB',
                 year=2019,
                 transform=None,
                 rgb=True,
                 rand_gap=3,
                 seq_len=3,
                 rand_reverse=True,
                 dynamic_merge=True,
                 enable_prev_frame=False,
                 max_obj_n=10,
                 merge_prob=0.3,
                 balance_sample=False,
                 balance_ratio=0.0,
                 use_vosp=False,
                 repeat=1):
        root = os.path.join(root, str(year), 'train')
        image_root = os.path.join(root, 'JPEGImages')
        label_root = os.path.join(root, 'Annotations')
        self.seq_list_file = os.path.join(root, 'meta.json')
        self._check_preprocess()
        seq_names = list(self.ann_f.keys())

        imglistdic = {}
        for seq_name in seq_names:
            data = self.ann_f[seq_name]['objects']
            obj_names = list(data.keys())
            images = []
            labels = []
            for obj_n in obj_names:
                if len(data[obj_n]["frames"]) < 2:
                    print("Short object: " + seq_name + '-' + obj_n)
                    continue
                images += list(
                    map(lambda x: x + '.jpg', list(data[obj_n]["frames"])))
                labels += list(
                    map(lambda x: x + '.png', list(data[obj_n]["frames"])))
            images = np.sort(np.unique(images))
            labels = np.sort(np.unique(labels))
            if len(images) < 2:
                print("Short video: " + seq_name)
                continue
            imglistdic[seq_name] = (images, labels)
        
        if balance_sample:
            import dataloaders.freq as freq
            if use_vosp:
                freq_dict = freq.VOSP_freq
            else:
                freq_dict = freq.VOS19_freq
        else:
            freq_dict = None

        super(YOUTUBEVOS_Train, self).__init__(image_root,
                                               label_root,
                                               imglistdic,
                                               transform,
                                               rgb,
                                               repeat,
                                               rand_gap,
                                               seq_len,
                                               rand_reverse,
                                               dynamic_merge,
                                               enable_prev_frame,
                                               merge_prob=merge_prob,
                                               max_obj_n=max_obj_n,
                                               balance_sample=balance_sample,
                                               balance_ratio=balance_ratio,
                                               freq_dict=freq_dict)

    def _check_preprocess(self):
        if not os.path.isfile(self.seq_list_file):
            print('No such file: {}.'.format(self.seq_list_file))
            return False
        else:
            self.ann_f = json.load(open(self.seq_list_file, 'r'))['videos']
            return True

class BL30K_Train(VOSTrain):
    def __init__(self,
                 root='./datasets/BL30K',
                 transform=None,
                 rgb=True,
                 rand_gap=12,
                 seq_len=3,
                 rand_reverse=True,
                 dynamic_merge=True,
                 enable_prev_frame=False,
                 max_obj_n=10,
                 merge_prob=0.3):
        image_root = os.path.join(root, 'JPEGImages')
        label_root = os.path.join(root, 'Annotations')
        seq_names = list(np.sort(os.listdir(os.path.join(image_root))))
        print('%d seqs in BL30K'%len(seq_names))

        imglistdic = {}
        for seq_name in seq_names:
            images = list(
                np.sort(os.listdir(os.path.join(image_root, seq_name))))
            labels = list(
                np.sort(os.listdir(os.path.join(label_root, seq_name))))
            if len(images) != len(labels):
                print(seq_name,'bad data!')
            else:
                imglistdic[seq_name] = (images, labels)

        super().__init__(image_root,
                            label_root,
                            imglistdic,
                            transform,
                            rgb,
                            1,
                            rand_gap,
                            seq_len,
                            rand_reverse,
                            dynamic_merge,
                            enable_prev_frame,
                            merge_prob=merge_prob,
                            max_obj_n=max_obj_n)

class VIPOSeg_Train(VOSTrain):
    def __init__(self,
                 root='./datasets/VIPOSeg',
                 transform=None,
                 rgb=True,
                 rand_gap=3,
                 seq_len=3,
                 rand_reverse=True,
                 dynamic_merge=True,
                 merge_prob=0.3,
                 enable_prev_frame=False,
                 train_pano=False,
                 max_stuff_num=6,
                 max_thing_num=10):
        # image_root = os.path.join(root, 'images')
        image_root = os.path.join(root, 'train/JPEGImages')
        label_root = os.path.join(root, 'train/Annotations')
        obj_class_file = os.path.join(root,'train/obj_class.json')
        # seq_names = list(np.sort(os.listdir(os.path.join(image_root))))
        seq_names = [x.split('/')[-1] for x in sorted(glob(image_root+'/*'))]
        self.train_pano = train_pano
        print('%d seqs in VIPSeg'%len(seq_names))
        import json
        with open(obj_class_file,'r') as f:
            self.obj_class_dict = json.load(f)
        print('VIPOSeg obj class file loaded')
        
        imglistdic = {}
        for seq_name in seq_names:
            images = list(
                np.sort(os.listdir(os.path.join(image_root, seq_name))))
            labels = list(
                np.sort(os.listdir(os.path.join(label_root, seq_name))))
            if len(images) != len(labels):
                print(seq_name,'bad data!')
            else:
                imglistdic[seq_name] = (images, labels)
        
        self.max_stuff_num = max_stuff_num
        self.max_thing_num = max_thing_num
        self.thing_class = [2, 4, 8, 10, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 60, 61, 62, 63,
         64, 65, 72, 74, 76, 77, 78, 79, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 95, 96, 97, 99, 100,
          101, 102, 106, 107, 108, 109, 114, 115, 116, 117, 118, 122, 123]
        self.stuff_class = [0, 1, 3, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
         26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 53, 57, 58, 59, 66, 67, 68, 69,
          70, 71, 73, 75, 80, 81, 93, 94, 98, 103, 104, 105, 110, 111, 112, 113, 119, 120, 121]

        super().__init__(image_root,
                            label_root,
                            imglistdic,
                            transform,
                            rgb,
                            1,
                            rand_gap,
                            seq_len,
                            rand_reverse,
                            dynamic_merge,
                            enable_prev_frame,
                            merge_prob=merge_prob,
                            max_obj_n=max_stuff_num + max_thing_num)
    
    def sample_sequence(self, idx):
        if not self.train_pano:
            return super().sample_sequence(idx)
        
        idx = idx % len(self.seqs)
        seqname = self.seqs[idx]
        seq_class_dict = self.obj_class_dict[seqname]
        imagelist, lablist = self.imglistdic[seqname]
        frame_num = len(imagelist)
        if self.rand_reverse:
            imagelist, lablist = self.reverse_seq(imagelist, lablist)

        is_consistent = False
        max_try = 5
        try_step = 0
        while (is_consistent is False and try_step < max_try):
            try_step += 1

            # generate random gaps
            curr_gaps, total_gap = self.get_curr_gaps(self.seq_len - 1)

            if self.enable_prev_frame:  # prev frame is randomly sampled
                # get prev frame
                prev_index = self.get_prev_index(lablist, total_gap)
                prev_image, prev_label = self.get_image_label(
                    seqname, imagelist, lablist, prev_index)
                prev_objs = list(np.unique(prev_label))

                # get curr frames
                curr_indices = self.get_curr_indices(lablist, prev_index,
                                                     curr_gaps)
                curr_images, curr_labels, curr_objs = [], [], []
                for curr_index in curr_indices:
                    curr_image, curr_label = self.get_image_label(
                        seqname, imagelist, lablist, curr_index)
                    c_objs = list(np.unique(curr_label))
                    curr_images.append(curr_image)
                    curr_labels.append(curr_label)
                    curr_objs.extend(c_objs)

                objs = list(np.unique(prev_objs + curr_objs))

                start_index = prev_index
                end_index = max(curr_indices)
                # get ref frame
                _try_step = 0
                ref_index = self.get_ref_index_v2(seqname, lablist)
                while (ref_index > start_index and ref_index <= end_index
                       and _try_step < max_try):
                    _try_step += 1
                    ref_index = self.get_ref_index_v2(seqname, lablist)
                ref_image, ref_label = self.get_image_label(
                    seqname, imagelist, lablist, ref_index)
                ref_objs = list(np.unique(ref_label))
            else:  # prev frame is next to ref frame
                # get ref frame
                ref_index = self.get_ref_index_v2(seqname, lablist)

                ref_image, ref_label = self.get_image_label(
                    seqname, imagelist, lablist, ref_index)
                ref_objs = list(np.unique(ref_label))

                # get curr frames
                curr_indices = self.get_curr_indices(lablist, ref_index,
                                                     curr_gaps)
                curr_images, curr_labels, curr_objs = [], [], []
                for curr_index in curr_indices:
                    curr_image, curr_label = self.get_image_label(
                        seqname, imagelist, lablist, curr_index)
                    c_objs = list(np.unique(curr_label))
                    curr_images.append(curr_image)
                    curr_labels.append(curr_label)
                    curr_objs.extend(c_objs)

                objs = list(np.unique(curr_objs))
                prev_image, prev_label = curr_images[0], curr_labels[0]
                curr_images, curr_labels = curr_images[1:], curr_labels[1:]

            is_consistent = True
            for obj in objs:
                if obj == 0:
                    continue
                if obj not in ref_objs:
                    is_consistent = False
                    break

        # get meta info
        obj_num = list(np.sort(ref_objs))[-1]
        
        # convert obj to thing and stuff
        obj_mapping_dict = {}
        thing_objs = []
        stuff_objs = []
        for obj in list(np.sort(ref_objs)):
            if obj==0:
                obj_mapping_dict[0] = [0,len(stuff_objs)]
                stuff_objs.append(obj)
            elif int(seq_class_dict[str(obj)]) in self.stuff_class:
                obj_mapping_dict[obj] = [0,len(stuff_objs)]
                stuff_objs.append(obj)
            elif int(seq_class_dict[str(obj)]) in self.thing_class:
                obj_mapping_dict[obj] = [1,len(thing_objs)]
                thing_objs.append(obj)
            else:
                raise ValueError("bad obj idx: {} in seq{}".format(obj,seqname))
        
        # restrict obj number in balanced random crop, not here
        # # check obj number
        # removed_objs = []
        # if len(stuff_objs) > self.max_stuff_num:
        #     random.shuffle(stuff_objs)
        #     sel_stuff_objs = sorted(stuff_objs[:self.max_stuff_num])
        #     removed_objs.extend(stuff_objs[self.max_stuff_num:])
        # if len(thing_objs) > self.max_thing_num:
        #     random.shuffle(thing_objs)
        #     sel_thing_objs = sorted(thing_objs[:self.max_thing_num])
        #     removed_objs.extend(thing_objs[self.max_thing_num:])
        
        # # remove extra objects in label
        # for obj in removed_objs:
        #     ref_label[ref_label==obj] = 0
        #     prev_label[prev_label==obj] = 0
        #     for i in range(len(curr_labels)):
        #         curr_labels[i][curr_labels[i]==obj] = 0
        
        obj_num = [len(stuff_objs),len(thing_objs)]


        sample = {
            'ref_img': ref_image,
            'prev_img': prev_image,
            'curr_img': curr_images,
            'ref_label': ref_label,
            'prev_label': prev_label,
            'curr_label': curr_labels
        }
        sample['meta'] = {
            'seq_name': seqname,
            'frame_num': frame_num,
            'obj_num': obj_num,
            'obj_mapping':obj_mapping_dict
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
        
class TEST(Dataset):
    def __init__(
        self,
        seq_len=3,
        obj_num=3,
        transform=None,
    ):
        self.seq_len = seq_len
        self.obj_num = obj_num
        self.transform = transform

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        img = np.zeros((800, 800, 3)).astype(np.float32)
        label = np.ones((800, 800)).astype(np.uint8)
        sample = {
            'ref_img': img,
            'prev_img': img,
            'curr_img': [img] * (self.seq_len - 2),
            'ref_label': label,
            'prev_label': label,
            'curr_label': [label] * (self.seq_len - 2)
        }
        sample['meta'] = {
            'seq_name': 'test',
            'frame_num': 100,
            'obj_num': self.obj_num
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
