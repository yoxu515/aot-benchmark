import os
import argparse
import numpy as np
import cv2
import multiprocessing
from shutil import copyfile
from glob import glob
from PIL import Image

try:
    from pycocotools.coco import COCO
except ImportError as e:
    print(e)

from utils.image import _palette as MASK_PALETTE

def get_args():
    parser = argparse.ArgumentParser(description='Unify Pretrain Dataset')
    parser.add_argument('--dst', type=str, default='./datasets/Static')
    parser.add_argument('--worker', type=int, default=10, help='Threads number.')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--src', type=str, required=True)

    # Custom dataset arguments
    parser.add_argument('--custom', action='store_true',
                        help='Use custom dataset mode. Requires --img_dir and --mask_dir.')
    parser.add_argument('--img_dir', type=str, default='images',
                        help='Image directory relative to --src (or absolute path).')
    parser.add_argument('--mask_dir', type=str, default='masks',
                        help='Mask directory relative to --src (or absolute path).')
    parser.add_argument('--img_ext', type=str, default='.jpg',
                        help='Image file extension, e.g. .jpg or .png')
    parser.add_argument('--mask_ext', type=str, default='.png',
                        help='Mask file extension.')
    parser.add_argument('--mask_format', type=str, default='binary',
                        choices=['binary', 'rgb_palette', 'index'],
                        help=(
                            'Mask format of your custom dataset:\n'
                            '  binary      -- single-object, pixel=255 is fg, pixel=0 is bg '
                            '(e.g. saliency datasets).\n'
                            '  rgb_palette -- multi-object, each object is a distinct RGB color '
                            '(e.g. MSRA10K-style).\n'
                            '  index       -- already uint8 object IDs (0=bg, 1,2,...=objects), '
                            'just copy as-is.'
                        ))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def cp_files(src_list, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for src_path in src_list:
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        copyfile(src_path, dst_path)


def _make_dst_dirs(dst_img_dir, dst_mask_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Mask conversion workers
# ---------------------------------------------------------------------------

def cvt_mask_palette(data):
    """RGB-color-per-object → uint8 object-ID palette PNG.
    Used by MSRA10K, ECSSD, PASCAL-S and custom rgb_palette datasets.
    """
    src_path, dst_dir = data

    mask = cv2.imread(src_path)
    mask_size = mask.shape[:2]

    label = np.asarray(mask).reshape(-1, 3)
    obj_labels = list(set(map(tuple, label)))
    obj_labels.sort()

    new_label = np.zeros(label.shape[0], np.uint8)

    obj_cnt = 0
    for idx, label_id in enumerate(obj_labels):
        tmp = int(label_id[0]) + int(label_id[1]) + int(label_id[2])
        if 0 < tmp / 3 < 128:
            continue
        new_label[(label == label_id).all(axis=1)] = obj_cnt
        obj_cnt += 1

    new_label = Image.fromarray(new_label.reshape(mask_size))
    new_label.putpalette(MASK_PALETTE)

    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    new_label.save(dst_path)


def cvt_mask_binary(data):
    """Binary mask (0/255) → uint8 object-ID PNG (0=bg, 1=fg).
    Used by single-object saliency-style custom datasets.
    No palette dependency — StaticTrain only needs correct uint8 IDs.
    """
    src_path, dst_path = data

    mask = np.array(Image.open(src_path).convert('L'), dtype=np.uint8)
    mask = (mask > 127).astype(np.uint8)   # 255 → 1, everything else → 0
    Image.fromarray(mask).save(dst_path)


def cvt_mask_index(data):
    """Already uint8 object-ID mask → just copy.
    Pixel values are already 0=bg, 1,2,...=object IDs.
    """
    src_path, dst_path = data
    copyfile(src_path, dst_path)


def cvt_mask_palette_VOC(data):
    """VOC2012-specific: strip void label (>20) then apply palette."""
    src_path, dst_path = data
    with Image.open(src_path) as img:
        mask = np.array(img.convert('P'))
    mask[mask > 20] = 0
    mask = Image.fromarray(mask)
    mask.putpalette(MASK_PALETTE)
    mask.save(dst_path)


# ---------------------------------------------------------------------------
# Per-dataset converters
# ---------------------------------------------------------------------------

def cvt_MSRA10K():
    img_list = sorted(glob(os.path.join(args.src, 'MSRA10K_Imgs_GT/Imgs/', '*.jpg')),
                      key=lambda x: (len(x), x))
    mask_list = sorted(glob(os.path.join(args.src, 'MSRA10K_Imgs_GT/Imgs/', '*.png')),
                       key=lambda x: (len(x), x))

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    cp_files(img_list, dst_img_dir)

    os.makedirs(dst_mask_dir, exist_ok=True)
    mask_list = [(x, dst_mask_dir) for x in mask_list]
    pools = multiprocessing.Pool(16)
    pools.map(cvt_mask_palette, mask_list)
    pools.close()
    pools.join()


def cvt_ECSSD():
    img_list = sorted(glob(os.path.join(args.src, 'images', '*.jpg')),
                      key=lambda x: (len(x), x))
    mask_list = sorted(glob(os.path.join(args.src, 'ground_truth_mask', '*.png')),
                       key=lambda x: (len(x), x))

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    cp_files(img_list, dst_img_dir)

    os.makedirs(dst_mask_dir, exist_ok=True)
    mask_list = [(x, dst_mask_dir) for x in mask_list]
    pools = multiprocessing.Pool(worker_n)
    pools.map(cvt_mask_palette, mask_list)
    pools.close()
    pools.join()


def cvt_PASCALS():
    img_list = sorted(glob(os.path.join(args.src, 'datasets/imgs/pascal', '*.jpg')),
                      key=lambda x: (len(x), x))
    mask_list = sorted(glob(os.path.join(args.src, 'datasets/masks/pascal', '*.png')),
                       key=lambda x: (len(x), x))

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    cp_files(img_list, dst_img_dir)

    os.makedirs(dst_mask_dir, exist_ok=True)
    mask_list = [(x, dst_mask_dir) for x in mask_list]
    pools = multiprocessing.Pool(worker_n)
    pools.map(cvt_mask_palette, mask_list)
    pools.close()
    pools.join()


def create_COCO_img_mask(data):
    img_id, dst_img_dir, dst_mask_dir = data

    img_info = coco.loadImgs(img_id)[0]
    h = img_info['height']
    w = img_info['width']

    mask_all = np.zeros((h, w), np.uint8)
    anno_ids = coco.getAnnIds(imgIds=img_info['id'])
    anno_list = coco.loadAnns(anno_ids)

    obj_cnt = 1
    for idx, anno in enumerate(anno_list):
        if anno['area'] < 500:
            continue
        mask = coco.annToMask(anno)
        mask_all[mask > 0] = mask[mask > 0] * obj_cnt
        obj_cnt += 1

    if obj_cnt > 1:
        mask_all = Image.fromarray(mask_all)
        mask_all.putpalette(MASK_PALETTE)

        img_name = img_info['file_name'][:-4] + '.png'
        dst_path = os.path.join(dst_mask_dir, img_name)
        mask_all.save(dst_path)

        tmp = img_info['coco_url'].split('/')[-2:]
        img_path_src = os.path.join(args.src, tmp[0], tmp[1])
        img_path_dst = os.path.join(dst_img_dir, img_info['file_name'])
        copyfile(img_path_src, img_path_dst)


def cvt_COCO():
    global coco

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    _make_dst_dirs(dst_img_dir, dst_mask_dir)

    for split in ('val2017', 'train2017'):
        anno_file = os.path.join(args.src, 'annotations', f'instances_{split}.json')
        print('Annotation path:', anno_file)
        coco = COCO(anno_file)
        img_list = [(x, dst_img_dir, dst_mask_dir) for x in coco.getImgIds()]
        pools = multiprocessing.Pool(worker_n)
        pools.imap(create_COCO_img_mask, img_list, chunk_size)
        pools.close()
        pools.join()


def cvt_VOC2012():
    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)

    img_set = os.path.join(args.src, 'ImageSets/Segmentation', 'trainval.txt')
    img_path_list = []
    mask_path_list = []
    with open(img_set, 'r') as lines:
        for line in lines:
            img_name = line.strip()
            img_path_list.append(os.path.join(args.src, 'JPEGImages', img_name + '.jpg'))
            dst_mask_path = os.path.join(dst_mask_dir, img_name + '.png')
            mask_path_list.append((
                os.path.join(args.src, 'SegmentationObject', img_name + '.png'),
                dst_mask_path
            ))

    os.makedirs(dst_mask_dir, exist_ok=True)
    pools = multiprocessing.Pool(worker_n)
    pools.imap(cvt_mask_palette_VOC, mask_path_list, chunk_size)
    pools.close()
    pools.join()

    cp_files(img_path_list, dst_img_dir)


# ---------------------------------------------------------------------------
# Custom dataset converter
# ---------------------------------------------------------------------------

def cvt_custom():
    """Normalize a custom dataset into the unified pretrain structure.

    Supports three mask formats:
      binary      -- single-object, pixel=255 fg / 0 bg (saliency-style)
      rgb_palette -- multi-object, each object is a distinct RGB color
      index       -- already uint8 object IDs, copied as-is

    Usage:
      python3 unify_pretrain_dataset.py \\
          --name MyDataset \\
          --src  /path/to/raw/dataset \\
          --dst  /path/to/unified \\
          --custom \\
          --img_dir  images \\
          --mask_dir masks \\
          --img_ext  .jpg \\
          --mask_format binary
    """
    # Resolve directories (support both relative-to-src and absolute paths)
    img_dir = args.img_dir if os.path.isabs(args.img_dir) \
        else os.path.join(args.src, args.img_dir)
    mask_dir = args.mask_dir if os.path.isabs(args.mask_dir) \
        else os.path.join(args.src, args.mask_dir)

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f'Image directory not found: {img_dir}')
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f'Mask directory not found: {mask_dir}')

    img_list = sorted(glob(os.path.join(img_dir, f'*{args.img_ext}')),
                      key=lambda x: (len(x), x))
    mask_list = sorted(glob(os.path.join(mask_dir, f'*{args.mask_ext}')),
                       key=lambda x: (len(x), x))

    if len(img_list) == 0:
        raise RuntimeError(f'No images found in {img_dir} with extension {args.img_ext}')
    if len(mask_list) == 0:
        raise RuntimeError(f'No masks found in {mask_dir} with extension {args.mask_ext}')
    if len(img_list) != len(mask_list):
        raise RuntimeError(
            f'Image/mask count mismatch: {len(img_list)} imgs vs {len(mask_list)} masks.\n'
            f'Ensure every image has a corresponding mask with the same stem.'
        )

    print(f'Custom dataset [{args.name}]: {len(img_list)} samples, '
          f'mask_format={args.mask_format}')

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    _make_dst_dirs(dst_img_dir, dst_mask_dir)

    # Images are always just copied
    cp_files(img_list, dst_img_dir)

    # Masks — route based on format
    if args.mask_format == 'binary':
        # (src_path, dst_path) pairs — binary worker needs per-file dst path
        pairs = [
            (src, os.path.join(dst_mask_dir,
                               os.path.splitext(os.path.basename(src))[0] + '.png'))
            for src in mask_list
        ]
        pools = multiprocessing.Pool(worker_n)
        pools.map(cvt_mask_binary, pairs)
        pools.close()
        pools.join()

    elif args.mask_format == 'rgb_palette':
        # reuse existing RGB→object-ID converter, needs mask_palette global
        pairs = [(src, dst_mask_dir) for src in mask_list]
        pools = multiprocessing.Pool(worker_n)
        pools.map(cvt_mask_palette, pairs)
        pools.close()
        pools.join()

    elif args.mask_format == 'index':
        # already correct uint8 IDs, straight copy
        pairs = [
            (src, os.path.join(dst_mask_dir, os.path.basename(src)))
            for src in mask_list
        ]
        pools = multiprocessing.Pool(worker_n)
        pools.map(cvt_mask_index, pairs)
        pools.close()
        pools.join()

    print(f'Done. Output at:\n'
          f'  JPEGImages  -> {dst_img_dir}\n'
          f'  Annotations -> {dst_mask_dir}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = get_args()
    worker_n = args.worker
    chunk_size = 100
    coco = None

    if args.custom:
        # Custom dataset — --name and --src still required, format via --mask_format
        cvt_custom()

    # Ming-Ming Cheng et al. Global contrast based salient region detection.
    # Download: https://mmcheng.net/msra10k/
    elif args.name == 'MSRA10K':
        cvt_MSRA10K()

    # Jianping Shi et al. Hierarchical image saliency detection on extended CSSD.
    # Download: http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html
    elif args.name == 'ECSSD':
        cvt_ECSSD()

    # Yin Li et al. The secrets of salient object segmentation.
    # Download: http://cbs.ic.gatech.edu/salobj
    elif args.name == 'PASCAL-S':
        cvt_PASCALS()

    # Bharath Hariharan et al. Semantic contours from inverse detectors.
    # Download: http://cocodataset.org/#download
    # pycocotools: https://github.com/cocodataset/cocoapi/tree/master/PythonAPI
    elif args.name == 'COCO':
        cvt_COCO()

    # Mark Everingham et al. The PASCAL Visual Object Classes (VOC) Challenge.
    # Download: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
    elif args.name == 'PASCALVOC2012':
        cvt_VOC2012()

    else:
        raise ValueError(
            f'Unknown dataset name: {args.name!r}.\n'
            f'Supported: MSRA10K, ECSSD, PASCAL-S, COCO, PASCALVOC2012.\n'
            f'For a custom dataset, pass --custom along with --img_dir, --mask_dir, --mask_format.'
        )
