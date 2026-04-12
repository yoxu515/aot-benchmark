import os
import argparse
import numpy as np
import cv2
import multiprocessing as mp
from shutil import copyfile
from glob import glob
from tqdm import tqdm
from PIL import Image

# =========================
# Unified Helpers
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def cp_files(src_list, dst_dir):
    ensure_dir(dst_dir)
    for src_path in tqdm(src_list, desc='Copying images'):
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        copyfile(src_path, dst_path)


# =========================
# Mask Converters (Unified API)
# Always: (src_path, dst_path)
# =========================

def mask_worker(args):
    src, dst, fn, palette = args
    fn(src, dst, palette)

def convert_mask_palette(src_path, dst_path, palette=None):
    mask = cv2.imread(src_path)
    h, w = mask.shape[:2]

    label = mask.reshape(-1, 3)
    unique_colors = list(set(map(tuple, label)))
    unique_colors.sort()

    new_mask = np.zeros(label.shape[0], np.uint8)
    obj_id = 0

    for color in unique_colors:
        avg = sum(color) / 3
        if 0 < avg < 128:
            continue
        new_mask[(label == color).all(axis=1)] = obj_id
        obj_id += 1

    new_mask = new_mask.reshape(h, w)
    new_mask = Image.fromarray(new_mask)

    if palette is not None:
        new_mask.putpalette(palette)

    new_mask.save(dst_path)

def convert_mask_voc(src_path, dst_path, palette=None):
    mask = np.array(Image.open(src_path).convert('P'))
    mask[mask > 20] = 0

    mask = Image.fromarray(mask)

    if palette is not None:
        mask.putpalette(palette)

    mask.save(dst_path)


# =========================
# Dataset Pipelines
# =========================

def process_masks(mask_paths, dst_dir, fn, palette, workers):
    
    ensure_dir(dst_dir)
    tasks = [
        (src, os.path.join(dst_dir, os.path.basename(src)), fn, palette)
        for src in mask_paths
    ]
    with mp.Pool(workers) as pool:
        list(tqdm(pool.imap(mask_worker, tasks), total=len(tasks)))


def cvt_MSRA10K(args, palette):
    img_paths = sorted(glob(os.path.join(args.src, 'Imgs/*.jpg')))
    mask_paths = sorted(glob(os.path.join(args.src, 'Imgs/*.png')))

    img_dst = os.path.join(args.dst, 'JPEGImages', args.name)
    mask_dst = os.path.join(args.dst, 'Annotations', args.name)

    cp_files(img_paths, img_dst)
    process_masks(mask_paths, mask_dst, convert_mask_palette, palette, args.worker)


def cvt_ECSSD(args, palette):
    img_paths = sorted(glob(os.path.join(args.src, 'images/*.jpg')))
    mask_paths = sorted(glob(os.path.join(args.src, 'ground_truth_mask/*.png')))

    img_dst = os.path.join(args.dst, 'JPEGImages', args.name)
    mask_dst = os.path.join(args.dst, 'Annotations', args.name)

    cp_files(img_paths, img_dst)
    process_masks(mask_paths, mask_dst, convert_mask_palette, palette, args.worker)


def cvt_PASCAL_S(args, palette):
    img_paths = sorted(glob(os.path.join(args.src, 'datasets/imgs/pascal/*.jpg')))
    mask_paths = sorted(glob(os.path.join(args.src, 'datasets/masks/pascal/*.png')))

    img_dst = os.path.join(args.dst, 'JPEGImages', args.name)
    mask_dst = os.path.join(args.dst, 'Annotations', args.name)

    cp_files(img_paths, img_dst)
    process_masks(mask_paths, mask_dst, convert_mask_palette, palette, args.worker)


def cvt_VOC2012(args, palette):
    img_dst = os.path.join(args.dst, 'JPEGImages', args.name)
    mask_dst = os.path.join(args.dst, 'Annotations', args.name)

    ensure_dir(mask_dst)

    img_set = os.path.join(args.src, 'ImageSets/Segmentation/trainval.txt')

    img_paths = []
    tasks = []

    with open(img_set, 'r') as f:
        for line in f:
            name = line.strip()

            img_paths.append(os.path.join(args.src, 'JPEGImages', name + '.jpg'))

            src_mask = os.path.join(args.src, 'SegmentationObject', name + '.png')
            dst_mask = os.path.join(mask_dst, name + '.png')

            tasks.append((src_mask, dst_mask))

    tasks = [
        (src, dst, convert_mask_voc, palette)
        for (src, dst) in tasks
    ]

    with mp.Pool(args.worker) as pool:
        list(tqdm(pool.imap(mask_worker, tasks), total=len(tasks)))

    cp_files(img_paths, img_dst)


# =========================
# Main
# =========================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', default='./datasets/Static')
    parser.add_argument('--name', required=True)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--palette', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    palette = None
    if args.palette is not None and os.path.exists(args.palette):
        palette = Image.open(args.palette).getpalette()

    if args.name == 'MSRA10K':
        cvt_MSRA10K(args, palette)

    elif args.name == 'ECSSD':
        cvt_ECSSD(args, palette)

    elif args.name == 'PASCAL-S':
        cvt_PASCAL_S(args, palette)

    elif args.name == 'PASCALVOC2012':
        cvt_VOC2012(args, palette)

    else:
        raise ValueError(f'Unknown dataset: {args.name}')
