import os
import time
import datetime as datetime
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.eval_datasets import YOUTUBEVOS_Test, YOUTUBEVOS_DenseTest, DAVIS_Test, EVAL_TEST
import dataloaders.video_transforms as tr

from utils.image import flip_tensor, save_mask
from utils.checkpoint import load_network
from utils.eval import zip_folder

from networks.models import build_vos_model
from networks.engines import build_engine

class Evaluator(object):
    def __init__(self, cfg, rank=0, seq_queue=None, info_queue=None):
        self.rank = rank
        self.cfg = cfg
        self.seq_queue = seq_queue
        self.info_queue = info_queue

        # --- Device setup ---
        # If TEST_GPU_ID is an int (0 or 1) and CUDA is available, use that GPU.
        # Otherwise fall back to CPU transparently.
        # --- Device setup ---
        if (
            isinstance(cfg.TEST_GPU_ID, int)
            and cfg.TEST_GPU_ID >= 0   
            and torch.cuda.is_available()
        ):
            self.gpu = cfg.TEST_GPU_ID + rank
            self.device = torch.device(f'cuda:{self.gpu}')
            torch.cuda.set_device(self.gpu)
            self.use_cuda = True
        else:
            self.gpu = None
            self.device = torch.device('cpu')
            self.use_cuda = False

        self.gpu_num = cfg.TEST_GPU_NUM

        self.print_log("Exp {}:".format(cfg.EXP_NAME))
        self.print_log(json.dumps(cfg.__dict__, indent=4, sort_keys=True))
        print("Using device: {}".format(self.device))

        self.print_log('Build VOS model.')
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).to(self.device)

        self.process_pretrained_model()
        self.prepare_dataset()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_device(self, tensor, non_blocking=False):
        """Move a tensor to the current device."""
        if self.use_cuda:
            return tensor.cuda(self.gpu, non_blocking=non_blocking)
        return tensor.to(self.device)

    def _new_timer(self):
        """
        Return a CUDA event on GPU, or a plain wall-clock object on CPU.
        Both share the same .record() interface; elapsed time is handled
        in _elapsed_seconds().
        """
        if self.use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return _CPUTimer()

    def _elapsed_seconds(self, start, end):
        """Return elapsed time in seconds between two timer objects."""
        if self.use_cuda:
            torch.cuda.synchronize()
            return start.elapsed_time(end) / 1e3
        return end.t - start.t

    def _max_mem_gb(self):
        """Peak GPU memory in GB, or 0.0 on CPU."""
        if self.use_cuda:
            return torch.cuda.max_memory_allocated(device=self.device) / (1024.0 ** 3)
        return 0.0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def process_pretrained_model(self):
        cfg = self.cfg

        if cfg.TEST_CKPT_PATH == 'test':
            self.ckpt = 'test'
            self.print_log('Test evaluation.')
            return

        if cfg.TEST_CKPT_PATH is None:
            if cfg.TEST_CKPT_STEP is not None:
                ckpt = str(cfg.TEST_CKPT_STEP)
            else:
                ckpts = os.listdir(cfg.DIR_CKPT)
                if len(ckpts) > 0:
                    ckpts = list(
                        map(lambda x: int(x.split('_')[-1].split('.')[0]), ckpts)
                    )
                    ckpt = np.sort(ckpts)[-1]
                else:
                    self.print_log('No checkpoint in {}.'.format(cfg.DIR_CKPT))
                    exit()

            self.ckpt = ckpt
            if cfg.TEST_EMA:
                cfg.DIR_CKPT = os.path.join(cfg.DIR_RESULT, 'ema_ckpt')
            cfg.TEST_CKPT_PATH = os.path.join(cfg.DIR_CKPT, 'save_step_%s.pth' % ckpt)

            try:
                self.model, removed_dict = load_network(self.model, cfg.TEST_CKPT_PATH, self.device)
            except Exception as inst:
                self.print_log(inst)
                self.print_log('Try to use backup checkpoint.')
                DIR_RESULT = './backup/{}/{}'.format(cfg.EXP_NAME, cfg.STAGE_NAME)
                DIR_CKPT = os.path.join(DIR_RESULT, 'ema_ckpt')
                TEST_CKPT_PATH = os.path.join(DIR_CKPT, 'save_step_%s.pth' % ckpt)
                self.model, removed_dict = load_network(self.model, TEST_CKPT_PATH, self.device)

            if len(removed_dict) > 0:
                self.print_log('Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load latest checkpoint from {}'.format(cfg.TEST_CKPT_PATH))

        else:
            self.ckpt = 'unknown'
            self.model, removed_dict = load_network(self.model, cfg.TEST_CKPT_PATH, self.device)
            if len(removed_dict) > 0:
                self.print_log('Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load checkpoint from {}'.format(cfg.TEST_CKPT_PATH))

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(
                cfg.TEST_MAX_SHORT_EDGE, cfg.TEST_MAX_LONG_EDGE,
                cfg.TEST_FLIP, cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS
            ),
            tr.MultiToTensor()
        ])

        exp_name = cfg.EXP_NAME
        if 'aost' in cfg.MODEL_VOS:
            exp_name += '_L{}'.format(int(cfg.MODEL_LSTT_NUM))

        eval_name = '{}_{}_{}_{}_ckpt_{}'.format(
            cfg.TEST_DATASET, cfg.TEST_DATASET_SPLIT,
            exp_name, cfg.STAGE_NAME, self.ckpt
        )

        if cfg.TEST_EMA:
            eval_name += '_ema'
        if cfg.TEST_FLIP:
            eval_name += '_flip'
        if len(cfg.TEST_MULTISCALE) > 1:
            eval_name += '_ms_' + (
                str(cfg.TEST_MULTISCALE)
                .replace('.', 'dot').replace('[', '').replace(']', '').replace(', ', '_')
            )

        if 'youtubevos' in cfg.TEST_DATASET:
            year = int(cfg.TEST_DATASET[-4:])
            self.result_root = os.path.join(
                cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations'
            )
            if '_all_frames' in cfg.TEST_DATASET_SPLIT:
                split = cfg.TEST_DATASET_SPLIT.split('_')[0]
                youtubevos_test = YOUTUBEVOS_DenseTest
                self.result_root_sparse = os.path.join(
                    cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                    eval_name + '_sparse', 'Annotations'
                )
                self.zip_dir_sparse = os.path.join(
                    cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                    '{}_sparse.zip'.format(eval_name)
                )
            else:
                split = cfg.TEST_DATASET_SPLIT
                youtubevos_test = YOUTUBEVOS_Test

            self.dataset = youtubevos_test(
                root=cfg.DIR_YTB, year=year, split=split,
                transform=eval_transforms, result_root=self.result_root
            )

        elif cfg.TEST_DATASET == 'davis2017':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(
                cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations', resolution
            )
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT], root=cfg.DIR_DAVIS, year=2017,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root
            )

        elif cfg.TEST_DATASET == 'davis2016':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(
                cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations', resolution
            )
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT], root=cfg.DIR_DAVIS, year=2016,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root
            )

        elif cfg.TEST_DATASET == 'test':
            self.result_root = os.path.join(
                cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations'
            )
            self.dataset = EVAL_TEST(eval_transforms, self.result_root)

        else:
            self.print_log('Unknown dataset!')
            exit()

        self.print_log('Eval {} on {} {}:'.format(
            cfg.EXP_NAME, cfg.TEST_DATASET, cfg.TEST_DATASET_SPLIT
        ))
        self.source_folder = os.path.join(
            cfg.DIR_EVALUATION, cfg.TEST_DATASET, eval_name, 'Annotations'
        )
        self.zip_dir = os.path.join(
            cfg.DIR_EVALUATION, cfg.TEST_DATASET, '{}.zip'.format(eval_name)
        )
        if not os.path.exists(self.result_root):
            try:
                os.makedirs(self.result_root)
            except Exception as inst:
                self.print_log(inst)
                self.print_log('Failed to make dir: {}.'.format(self.result_root))
        self.print_log('Done!')

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    def evaluating(self):
        cfg = self.cfg
        self.model.eval()

        video_num = 0
        processed_video_num = 0
        total_time = 0
        total_frame = 0
        total_sfps = 0
        total_video_num = len(self.dataset)
        start_eval_time = time.time()

        if self.seq_queue is not None:
            if self.rank == 0:
                for seq_idx in range(total_video_num):
                    self.seq_queue.put(seq_idx)
                for _ in range(self.gpu_num):
                    self.seq_queue.put('END')
            coming_seq_idx = self.seq_queue.get()

        all_engines = []
        with torch.no_grad():
            for seq_idx, seq_dataset in enumerate(self.dataset):
                video_num += 1

                if self.seq_queue is not None:
                    if coming_seq_idx == 'END':
                        break
                    elif coming_seq_idx != seq_idx:
                        continue
                    else:
                        coming_seq_idx = self.seq_queue.get()

                processed_video_num += 1

                for engine in all_engines:
                    engine.restart_engine()

                seq_name = seq_dataset.seq_name
                print('Device {} - Processing Seq {} [{}/{}]:'.format(
                    self.device, seq_name, video_num, total_video_num
                ))
                if self.use_cuda:
                    torch.cuda.empty_cache()

                seq_dataloader = DataLoader(
                    seq_dataset, batch_size=1, shuffle=False,
                    num_workers=cfg.TEST_WORKERS, pin_memory=self.use_cuda
                )

                if 'all_frames' in cfg.TEST_DATASET_SPLIT:
                    images_sparse = seq_dataset.images_sparse
                    seq_dir_sparse = os.path.join(self.result_root_sparse, seq_name)
                    if not os.path.exists(seq_dir_sparse):
                        os.makedirs(seq_dir_sparse)

                seq_total_time = 0
                seq_total_frame = 0
                seq_pred_masks = {'dense': [], 'sparse': []}
                seq_timers = []

                for frame_idx, samples in enumerate(seq_dataloader):

                    all_preds = []
                    new_obj_label = None
                    aug_num = len(samples)

                    for aug_idx in range(aug_num):
                        if len(all_engines) <= aug_idx:
                            all_engines.append(
                                build_engine(
                                    cfg.MODEL_ENGINE,
                                    phase='eval',
                                    aot_model=self.model,
                                    gpu_id=self.gpu if self.use_cuda else None,
                                    long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
                                    short_term_mem_skip=cfg.TEST_SHORT_TERM_MEM_SKIP,
                                )
                            )
                            all_engines[-1].eval()

                        if aug_num > 1 and self.use_cuda:
                            torch.cuda.empty_cache()

                        engine = all_engines[aug_idx]
                        sample = samples[aug_idx]

                        is_flipped = sample['meta']['flip']
                        obj_nums = [int(n) for n in sample['meta']['obj_num']]
                        imgname = sample['meta']['current_name']
                        ori_height = sample['meta']['height']
                        ori_width = sample['meta']['width']
                        obj_idx = [int(i) for i in sample['meta']['obj_idx']]

                        current_img = self._to_device(sample['current_img'], non_blocking=True)
                        sample['current_img'] = current_img

                        current_label = None
                        if 'current_label' in sample:
                            current_label = self._to_device(
                                sample['current_label'], non_blocking=True
                            ).float()

                        # ---- per-frame logic ----
                        if frame_idx == 0:
                            _current_label = F.interpolate(
                                current_label, size=current_img.size()[2:], mode='nearest'
                            )
                            engine.add_reference_frame(
                                current_img, _current_label,
                                frame_step=0, obj_nums=obj_nums
                            )
                        else:
                            if aug_idx == 0:
                                seq_timers.append([self._new_timer()])

                            engine.match_propogate_one_frame(current_img)
                            pred_logit = engine.decode_current_logits((ori_height, ori_width))

                            if is_flipped:
                                pred_logit = flip_tensor(pred_logit, 3)

                            pred_prob = torch.softmax(pred_logit, dim=1)
                            all_preds.append(pred_prob)

                            if not is_flipped and current_label is not None and new_obj_label is None:
                                new_obj_label = current_label

                    if frame_idx > 0:
                        all_pred_probs = [
                            torch.mean(pred, dim=0, keepdim=True) for pred in all_preds
                        ]
                        all_pred_labels = [
                            torch.argmax(prob, dim=1, keepdim=True).float()
                            for prob in all_pred_probs
                        ]

                        cat_all_preds = torch.cat(all_preds, dim=0)
                        pred_prob = torch.mean(cat_all_preds, dim=0, keepdim=True)
                        pred_label = torch.argmax(pred_prob, dim=1, keepdim=True).float()

                        if new_obj_label is not None:
                            keep = (new_obj_label == 0).float()
                            all_pred_labels = [
                                label * keep + new_obj_label * (1 - keep)
                                for label in all_pred_labels
                            ]
                            pred_label = pred_label * keep + new_obj_label * (1 - keep)
                            new_obj_nums = [int(pred_label.max().item())]

                            if cfg.TEST_FLIP:
                                all_flip_pred_labels = [
                                    flip_tensor(label, 3) for label in all_pred_labels
                                ]

                            for aug_idx in range(len(samples)):
                                engine = all_engines[aug_idx]
                                current_img = samples[aug_idx]['current_img']
                                current_label = (
                                    all_flip_pred_labels[aug_idx]
                                    if samples[aug_idx]['meta']['flip']
                                    else all_pred_labels[aug_idx]
                                )
                                current_label = F.interpolate(
                                    current_label, size=engine.input_size_2d, mode='nearest'
                                )
                                engine.add_reference_frame(
                                    current_img, current_label,
                                    obj_nums=new_obj_nums, frame_step=frame_idx
                                )
                                engine.decode_current_logits((ori_height, ori_width))
                                engine.update_memory(current_label)

                        else:
                            if not cfg.MODEL_USE_PREV_PROB:
                                if cfg.TEST_FLIP:
                                    all_flip_pred_labels = [
                                        flip_tensor(label, 3) for label in all_pred_labels
                                    ]

                                for aug_idx in range(len(samples)):
                                    engine = all_engines[aug_idx]
                                    current_label = (
                                        all_flip_pred_labels[aug_idx]
                                        if samples[aug_idx]['meta']['flip']
                                        else all_pred_labels[aug_idx]
                                    )
                                    current_label = F.interpolate(
                                        current_label, size=engine.input_size_2d, mode='nearest'
                                    )
                                    engine.update_memory(current_label)
                            else:
                                if cfg.TEST_FLIP:
                                    all_flip_pred_probs = [
                                        flip_tensor(prob, 3) for prob in all_pred_probs
                                    ]

                                for aug_idx in range(len(samples)):
                                    engine = all_engines[aug_idx]
                                    current_prob = (
                                        all_flip_pred_probs[aug_idx]
                                        if samples[aug_idx]['meta']['flip']
                                        else all_pred_probs[aug_idx]
                                    )
                                    current_prob = F.interpolate(
                                        current_prob, size=engine.input_size_2d, mode='nearest'
                                    )
                                    engine.update_memory(current_prob)

                        seq_timers[-1].append(self._new_timer())

                        if cfg.TEST_FRAME_LOG:
                            one_frametime = self._elapsed_seconds(
                                seq_timers[-1][0], seq_timers[-1][1]
                            )
                            print('Device {} - Frame: {} - Obj Num: {}, Time: {}ms'.format(
                                self.device, imgname[0].split('.')[0],
                                obj_nums[0], int(one_frametime * 1e3)
                            ))

                        seq_pred_masks['dense'].append({
                            'path': os.path.join(
                                self.result_root, seq_name,
                                imgname[0].split('.')[0] + '.png'
                            ),
                            'mask': pred_label,
                            'obj_idx': obj_idx,
                        })
                        if 'all_frames' in cfg.TEST_DATASET_SPLIT and imgname in images_sparse:
                            seq_pred_masks['sparse'].append({
                                'path': os.path.join(
                                    self.result_root_sparse, seq_name,
                                    imgname[0].split('.')[0] + '.png'
                                ),
                                'mask': pred_label,
                                'obj_idx': obj_idx,
                            })

                # Save masks
                for mask_result in seq_pred_masks['dense'] + seq_pred_masks['sparse']:
                    save_mask(
                        mask_result['mask'].squeeze(0).squeeze(0),
                        mask_result['path'],
                        mask_result['obj_idx']
                    )
                del seq_pred_masks

                for timer in seq_timers:
                    one_frametime = self._elapsed_seconds(timer[0], timer[1])
                    seq_total_time += one_frametime
                    seq_total_frame += 1
                del seq_timers

                seq_avg_time_per_frame = seq_total_time / seq_total_frame
                total_time += seq_total_time
                total_frame += seq_total_frame
                total_avg_time_per_frame = total_time / total_frame
                total_sfps += seq_avg_time_per_frame
                avg_sfps = total_sfps / processed_video_num
                max_mem = self._max_mem_gb()

                print(
                    "Device {} - Seq {} - FPS: {:.2f}. "
                    "All-Frame FPS: {:.2f}, All-Seq FPS: {:.2f}, Max Mem: {:.2f}G".format(
                        self.device, seq_name,
                        1.0 / seq_avg_time_per_frame,
                        1.0 / total_avg_time_per_frame,
                        1.0 / avg_sfps,
                        max_mem,
                    )
                )

        if self.seq_queue is not None:
            if self.rank != 0:
                self.info_queue.put({
                    'total_time': total_time,
                    'total_frame': total_frame,
                    'total_sfps': total_sfps,
                    'processed_video_num': processed_video_num,
                    'max_mem': max_mem,
                })
            print('Finished the evaluation on device {}.'.format(self.device))

            if self.rank == 0:
                for _ in range(self.gpu_num - 1):
                    info_dict = self.info_queue.get()
                    total_time += info_dict['total_time']
                    total_frame += info_dict['total_frame']
                    total_sfps += info_dict['total_sfps']
                    processed_video_num += info_dict['processed_video_num']
                    max_mem = max(max_mem, info_dict['max_mem'])

                print(
                    "Devices {} - All-Frame FPS: {:.2f}, All-Seq FPS: {:.2f}, Max Mem: {:.2f}G".format(
                        list(range(self.gpu_num)),
                        1.0 / (total_time / total_frame),
                        1.0 / (total_sfps / processed_video_num),
                        max_mem,
                    )
                )
        else:
            print(
                "Device {} - All-Frame FPS: {:.2f}, All-Seq FPS: {:.2f}, Max Mem: {:.2f}G".format(
                    self.device,
                    1.0 / total_avg_time_per_frame,
                    1.0 / avg_sfps,
                    max_mem,
                )
            )

        if self.rank == 0:
            zip_folder(self.source_folder, self.zip_dir)
            self.print_log('Saving result to {}.'.format(self.zip_dir))
            if 'all_frames' in cfg.TEST_DATASET_SPLIT:
                zip_folder(self.result_root_sparse, self.zip_dir_sparse)
            total_eval_time = str(
                datetime.timedelta(seconds=int(time.time() - start_eval_time))
            )
            self.print_log("Total evaluation time: {}".format(total_eval_time))

    def print_log(self, string):
        if self.rank == 0:
            print(string)


# ---------------------------------------------------------------------------
# Minimal CPU wall-clock timer that mirrors the torch.cuda.Event interface
# ---------------------------------------------------------------------------

class _CPUTimer:
    """Lightweight wall-clock stand-in for torch.cuda.Event on CPU.

    Matches the two methods used by the Evaluator:
      - .record()          — snap the current time
      - .elapsed_time(end) — return ms elapsed up to `end` (CUDA convention)
    """

    def __init__(self):
        self.t = None
        self.record()  # record immediately on creation, same as CUDA events

    def record(self):
        self.t = time.perf_counter()

    def elapsed_time(self, end: '_CPUTimer') -> float:
        """Return elapsed time in milliseconds (matches torch.cuda.Event)."""
        return (end.t - self.t) * 1e3