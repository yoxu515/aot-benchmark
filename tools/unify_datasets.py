import os
import argparse
import json
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import shutil

# =========================
# Utilities
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_symlink_or_copy(src, dst):
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


# =========================
# Dataset Loader (SAFE)
# =========================

def load_dataset_safe(src_dir, img_dir, mask_dir):
    imgs = glob(os.path.join(src_dir, img_dir, '*'))
    masks = glob(os.path.join(src_dir, mask_dir, '*'))

    img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}

    keys = sorted(set(img_map.keys()) & set(mask_map.keys()))

    img_paths = [img_map[k] for k in keys]
    mask_paths = [mask_map[k] for k in keys]

    return img_paths, mask_paths


# =========================
# Global Color Mapping
# =========================

def build_global_color_map(mask_paths):
    color_set = set()

    for path in tqdm(mask_paths, desc='Scanning colors'):
        mask = np.array(Image.open(path).convert('RGB'))
        colors = np.unique(mask.reshape(-1, 3), axis=0)
        for c in colors:
            color_set.add(tuple(c))

    color_list = sorted(list(color_set))
    return {color: idx for idx, color in enumerate(color_list)}


# =========================
# Mask Conversion
# =========================

def convert_mask(mask_path, dst_path, color_map, palette=None):
    mask = np.array(Image.open(mask_path).convert('RGB'))
    h, w = mask.shape[:2]

    flat = mask.reshape(-1, 3)
    new_mask = np.zeros(flat.shape[0], dtype=np.uint8)

    for color, cls_id in color_map.items():
        matches = np.all(flat == color, axis=1)
        new_mask[matches] = cls_id

    new_mask = new_mask.reshape(h, w)

    img = Image.fromarray(new_mask)
    if palette is not None:
        img.putpalette(palette)

    img.save(dst_path)


# =========================
# Validation
# =========================

def validate_pair(img_path, mask_path):
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    if img.size != mask.size:
        raise ValueError(f"Size mismatch: {img_path} vs {mask_path}")


# =========================
# Processing
# =========================

def process_sample(args):
    img_path, mask_path, img_dst, mask_dst, color_map, palette = args

    validate_pair(img_path, mask_path)

    img_out = os.path.join(img_dst, os.path.basename(img_path))
    mask_out = os.path.join(mask_dst, os.path.basename(mask_path))

    safe_symlink_or_copy(img_path, img_out)
    convert_mask(mask_path, mask_out, color_map, palette)


# =========================
# Index Writer
# =========================

def write_index(img_paths, dst_dir):
    with open(os.path.join(dst_dir, 'train.txt'), 'w') as f:
        for p in img_paths:
            f.write(os.path.basename(p) + '\n')


# =========================
# Main Pipeline
# =========================

def run_pipeline(args):
    img_paths, mask_paths = load_dataset_safe(args.src, args.img_dir, args.mask_dir)

    img_dst = os.path.join(args.dst, 'JPEGImages', args.name)
    mask_dst = os.path.join(args.dst, 'Annotations', args.name)

    ensure_dir(img_dst)
    ensure_dir(mask_dst)

    # Build global mapping
    color_map = build_global_color_map(mask_paths)

    # Save mapping
    with open(os.path.join(args.dst, f'{args.name}_color_map.json'), 'w') as f:
        json.dump({str(k): v for k, v in color_map.items()}, f, indent=2)

    palette = None
    if args.palette and os.path.exists(args.palette):
        palette = Image.open(args.palette).getpalette()

    tasks = [
        (img_paths[i], mask_paths[i], img_dst, mask_dst, color_map, palette)
        for i in range(len(img_paths))
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(process_sample, tasks), total=len(tasks), desc='Processing'))

    write_index(img_paths, os.path.join(args.dst, args.name))

    # Save config
    with open(os.path.join(args.dst, f'{args.name}_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Delete source if requested
    if args.delete_src:
        shutil.rmtree(args.src)


# =========================
# CLI
# =========================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--img_dir', default='images')
    parser.add_argument('--mask_dir', default='masks')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--palette', default=None)
    parser.add_argument('--delete_src', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_pipeline(args)
