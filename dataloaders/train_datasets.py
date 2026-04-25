import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as TF

import dataloaders.image_transforms as IT


class StaticTrain(Dataset):
    def __init__(
        self,
        root,
        dataset_name,
        output_size,
        seq_len=5,
        max_obj_n=10,
        dynamic_merge=True,
        merge_prob=1.0,
        aug_type='v1'
    ):
        self.root = root
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.max_obj_n = max_obj_n
        self.dynamic_merge = dynamic_merge
        self.merge_prob = merge_prob

        # === Paths ===
        self.img_dir = os.path.join(root, 'JPEGImages', dataset_name)
        self.mask_dir = os.path.join(root, 'Annotations', dataset_name)
        self.index_file = os.path.join(root, dataset_name, 'train.txt')

        if not os.path.exists(self.index_file):
            raise RuntimeError(f"Missing index file: {self.index_file}")

        with open(self.index_file) as f:
            self.ids = [line.strip() for line in f]

        if len(self.ids) == 0:
            raise RuntimeError("Dataset is empty")

        print(f"[StaticTrain] {dataset_name}: {len(self.ids)} samples")

        # === Augmentations ===
        self.pre_flip = IT.RandomHorizontalFlip(0.5)
        self.flip = IT.RandomHorizontalFlip(0.3)

        if aug_type == 'v1':
            self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.03)
        elif aug_type == 'v2':
            self.color_jitter = TF.RandomApply(
                [TF.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            )
            self.gray = TF.RandomGrayscale(p=0.2)
            self.blur = TF.RandomApply([IT.GaussianBlur([.1, 2.])], p=0.3)
        else:
            raise ValueError("Unknown aug_type")

        base_ratio = float(output_size[1]) / output_size[0]

        self.affine = IT.RandomAffine(
            degrees=20,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10,
            resample=Image.BICUBIC,
            fillcolor=(124, 116, 104)
        )

        self.resize_crop = IT.RandomResizedCrop(
            output_size,
            (0.8, 1),
            ratio=(base_ratio * 3./4., base_ratio * 4./3.),
            interpolation=Image.BICUBIC
        )

        self.to_tensor = TF.ToTensor()
        self.normalize = TF.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )

        self.to_onehot = IT.ToOnehot(max_obj_n, shuffle=True)

    def __len__(self):
        return len(self.ids)

    # =========================
    # Loading
    # =========================

    def _load_pair(self, img_id):
        name = os.path.splitext(img_id)[0]

        img_path = os.path.join(self.img_dir, img_id)
        mask_path = os.path.join(self.mask_dir, name + '.png')

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise RuntimeError(f"Missing pair: {img_id}")

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if mask.mode != 'L':
            mask = mask.convert('L')

        if img.size != mask.size:
            raise RuntimeError(f"Size mismatch: {img_id}")

        return img, mask

    # =========================
    # Sampling
    # =========================

    def _sample_sequence(self, idx):
        img_id = self.ids[idx]
        img_pil, mask_pil = self._load_pair(img_id)

        frames, masks = [], []

        img_pil, mask_pil = self.pre_flip(img_pil, mask_pil)

        for i in range(self.seq_len):
            img, mask = img_pil, mask_pil

            if i > 0:
                img, mask = self.flip(img, mask)
                img, mask = self.affine(img, mask)

            img = self.color_jitter(img)
            img, mask = self.resize_crop(img, mask)

            mask = np.array(mask, np.uint8)

            # IMPORTANT: keep labels bounded
            mask = np.clip(mask, 0, self.max_obj_n)

            if i == 0:
                mask, obj_list = self.to_onehot(mask)
                obj_num = len(obj_list)

                # resample if useless
                if obj_num == 0:
                    return self._sample_sequence(random.randint(0, len(self.ids)-1))
            else:
                mask, _ = self.to_onehot(mask, obj_list)

            mask = torch.argmax(mask, dim=0, keepdim=True)

            frames.append(self.normalize(self.to_tensor(img)))
            masks.append(mask)

        return {
            'ref_img': frames[0],
            'prev_img': frames[1],
            'curr_img': frames[2:],
            'ref_label': masks[0],
            'prev_label': masks[1],
            'curr_label': masks[2:],
            'meta': {
                'id': img_id,
                'obj_num': obj_num
            }
        }

    # =========================
    # Main access
    # =========================

    def __getitem__(self, idx):
        for _ in range(5):  # retry safety
            try:
                sample1 = self._sample_sequence(idx)

                if self.dynamic_merge and (
                    sample1['meta']['obj_num'] > 0 and random.random() < self.merge_prob
                ):
                    rand_idx = random.randint(0, len(self.ids) - 1)
                    while rand_idx == idx:
                        rand_idx = random.randint(0, len(self.ids) - 1)

                    sample2 = self._sample_sequence(rand_idx)
                    return _merge_sample(sample1, sample2, max_obj_n=self.max_obj_n)

                return sample1

            except Exception:
                idx = random.randint(0, len(self.ids) - 1)

        raise RuntimeError("Too many failed samples")