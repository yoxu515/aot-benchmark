import os
import time
import datetime as datetime
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.eval_datasets import YOUTUBEVOS_Test, YOUTUBEVOS_DenseTest, DAVIS_Test, EVAL_TEST, VIPOSeg_Test
import dataloaders.video_transforms as tr

from utils.image import flip_tensor, save_mask, save_prob, save_logit, label2box, check_box, box_filter
from utils.checkpoint import load_network
from utils.eval import zip_folder

from networks.models import build_vos_model
from networks.engines import build_engine


class Evaluator(object):
    def __init__(self, cfg, rank=0, seq_queue=None, info_queue=None):
        self.gpu = cfg.TEST_GPU_ID + rank
        self.gpu_num = cfg.TEST_GPU_NUM
        self.rank = rank
        self.cfg = cfg
        self.seq_queue = seq_queue
        self.info_queue = info_queue

        self.print_log("Exp {}:".format(cfg.EXP_NAME))
        self.print_log(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

        print("Use GPU {} for evaluating.".format(self.gpu))
        torch.cuda.set_device(self.gpu)

        self.print_log('Build VOS model.')
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(self.gpu)

        self.process_pretrained_model()

        self.prepare_dataset()

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
                        map(lambda x: int(x.split('_')[-1].split('.')[0]),
                            ckpts))
                    ckpt = np.sort(ckpts)[-1]
                else:
                    self.print_log('No checkpoint in {}.'.format(cfg.DIR_CKPT))
                    exit()
            self.ckpt = ckpt
            if cfg.TEST_EMA:
                cfg.DIR_CKPT = os.path.join(cfg.DIR_RESULT, 'ema_ckpt')
            cfg.TEST_CKPT_PATH = os.path.join(cfg.DIR_CKPT,
                                              'save_step_%s.pth' % ckpt)
            self.model, removed_dict = load_network(self.model,
                                                    cfg.TEST_CKPT_PATH,
                                                    self.gpu)
            if len(removed_dict) > 0:
                self.print_log(
                    'Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load latest checkpoint from {}'.format(
                cfg.TEST_CKPT_PATH))
        else:
            if cfg.TEST_CKPT_STEP is not None:
                self.ckpt = str(cfg.TEST_CKPT_STEP)
            else:
                self.ckpt = 'unknown'
            self.model, removed_dict = load_network(self.model,
                                                    cfg.TEST_CKPT_PATH,
                                                    self.gpu)
            if len(removed_dict) > 0:
                self.print_log(
                    'Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load checkpoint from {}'.format(
                cfg.TEST_CKPT_PATH))

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MAX_SHORT_EDGE,
                                 cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP, cfg.TEST_INPLACE_FLIP,
                                 cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

        exp_name = cfg.EXP_NAME
        if 'aost' in cfg.MODEL_VOS:
            exp_name += '_L{}'.format(int(cfg.MODEL_LSTT_NUM))

        eval_name = '{}_{}_{}_{}_ckpt_{}'.format(cfg.TEST_DATASET,
                                                 cfg.TEST_DATASET_SPLIT,
                                                 exp_name, cfg.STAGE_NAME,
                                                 self.ckpt)

        if cfg.TEST_EMA:
            eval_name += '_ema'
        eval_name += '_gap' + str(cfg.TEST_LONG_TERM_MEM_GAP) + '-' + str(cfg.TEST_SHORT_TERM_MEM_GAP)
        if cfg.TEST_LONG_TERM_MEM_MAX < 999:
            eval_name += '_max' + str(cfg.TEST_LONG_TERM_MEM_MAX)
        if cfg.TEST_TOP_K != -1:
            if cfg.TEST_TOP_K>1:
                topk_s = str(int(cfg.TEST_TOP_K))
            else:
                topk_s = str(cfg.TEST_TOP_K).replace('.','-')
            eval_name += '_topk' + topk_s
        if cfg.TEST_INPLACE_FLIP:
            eval_name += '_inflip'
        elif cfg.TEST_FLIP:
            eval_name += '_flip'
        if cfg.TEST_MULTISCALE != [1.]:
            eval_name += '_ms_' + str(cfg.TEST_MULTISCALE).replace(
                '.', 'dot').replace('[', '').replace(']', '').replace(
                    ', ', '_')

        if 'youtubevos' in cfg.TEST_DATASET:
            year = int(cfg.TEST_DATASET[-4:])
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations')
            if cfg.TEST_SAVE_PROB:
                self.result_root_prob = os.path.join(cfg.DIR_EVALUATION,
                                                       cfg.TEST_DATASET,
                                                       eval_name + '_prob',
                                                       'Annotations')
            if cfg.TEST_SAVE_LOGIT:
                self.result_root_logit = os.path.join(cfg.DIR_EVALUATION,
                                                       cfg.TEST_DATASET,
                                                       eval_name + '_logit',
                                                       'Annotations')
            if '_all_frames' in cfg.TEST_DATASET_SPLIT:
                split = cfg.TEST_DATASET_SPLIT.split('_')[0]
                youtubevos_test = YOUTUBEVOS_DenseTest

                self.result_root_sparse = os.path.join(cfg.DIR_EVALUATION,
                                                       cfg.TEST_DATASET,
                                                       eval_name + '_sparse',
                                                       'Annotations')
                self.zip_dir_sparse = os.path.join(
                    cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                    '{}_sparse.zip'.format(eval_name))
            else:
                split = cfg.TEST_DATASET_SPLIT
                youtubevos_test = YOUTUBEVOS_Test

            self.dataset = youtubevos_test(root=cfg.DIR_YTB,
                                           year=year,
                                           split=split,
                                           transform=eval_transforms,
                                           result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2017':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations', resolution)
            if cfg.TEST_SAVE_PROB:
                self.result_root_prob = os.path.join(cfg.DIR_EVALUATION,
                                                       cfg.TEST_DATASET,
                                                       eval_name + '_prob',
                                                       'Annotations')
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_DAVIS,
                year=2017,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2016':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations', resolution)
            if cfg.TEST_SAVE_PROB:
                self.result_root_prob = os.path.join(cfg.DIR_EVALUATION,
                                                       cfg.TEST_DATASET,
                                                       eval_name + '_prob',
                                                       'Annotations')
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_DAVIS,
                year=2016,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'test':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations')
            self.dataset = EVAL_TEST(eval_transforms, self.result_root)
        elif cfg.TEST_DATASET == 'viposeg':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations')
            if cfg.TEST_SAVE_PROB:
                self.result_root_prob = os.path.join(cfg.DIR_EVALUATION,
                                                       cfg.TEST_DATASET,
                                                       eval_name + '_prob',
                                                       'Annotations')
            split = cfg.TEST_DATASET_SPLIT
            self.dataset = VIPOSeg_Test(root=cfg.DIR_VIP,
                                           split=split,
                                           transform=eval_transforms,
                                           result_root=self.result_root,
                                           test_pano=cfg.TEST_PANO)
        else:
            self.print_log('Unknown dataset!')
            exit()

        self.print_log('Eval {} on {} {}:'.format(cfg.EXP_NAME,
                                                  cfg.TEST_DATASET,
                                                  cfg.TEST_DATASET_SPLIT))
        self.source_folder = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                                          eval_name, 'Annotations')
        self.zip_dir = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                                    '{}.zip'.format(eval_name))
        if not os.path.exists(self.result_root):
            try:
                os.makedirs(self.result_root)
            except Exception as inst:
                self.print_log(inst)
                self.print_log('Failed to mask dir: {}.'.format(
                    self.result_root))
        self.print_log('Done!')

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

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

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
                print('GPU {} - Processing Seq {} [{}/{}]:'.format(
                    self.gpu, seq_name, video_num, total_video_num))
                torch.cuda.empty_cache()

                seq_dataloader = DataLoader(seq_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=cfg.TEST_WORKERS,
                                            pin_memory=True)

                if 'all_frames' in cfg.TEST_DATASET_SPLIT:
                    images_sparse = seq_dataset.images_sparse
                    seq_dir_sparse = os.path.join(self.result_root_sparse,
                                                  seq_name)
                    if not os.path.exists(seq_dir_sparse):
                        os.makedirs(seq_dir_sparse)
                if cfg.TEST_SAVE_PROB:
                    seq_dir_prob = os.path.join(self.result_root_prob,
                                                  seq_name)
                    if not os.path.exists(seq_dir_prob):
                        os.makedirs(seq_dir_prob)
                if cfg.TEST_SAVE_LOGIT:
                    seq_dir_logit = os.path.join(self.result_root_logit,
                                                  seq_name)
                    if not os.path.exists(seq_dir_logit):
                        os.makedirs(seq_dir_logit)

                seq_total_time = 0
                seq_total_frame = 0
                box_dict = {}

                for frame_idx, samples in enumerate(seq_dataloader):

                    all_preds = []
                    new_obj_label = None
                    one_frametime = 0
                    aug_num = len(samples)

                    for aug_idx in range(aug_num):
                        if len(all_engines) <= aug_idx:
                            all_engines.append(
                                build_engine(cfg.MODEL_ENGINE,
                                             phase='eval',
                                             aot_model=self.model,
                                             gpu_id=self.gpu,
                                             long_term_mem_gap=self.cfg.
                                             TEST_LONG_TERM_MEM_GAP,
                                             short_term_mem_skip=self.cfg.
                                             TEST_SHORT_TERM_MEM_GAP))
                            all_engines[-1].eval()

                        if aug_num > 1:  # if use test-time augmentation
                            torch.cuda.empty_cache()  # release GPU memory

                        engine = all_engines[aug_idx]

                        sample = samples[aug_idx]
                        # if sample['meta']['seq_name'] == ['0e4068b53f']:
                        #     print('get it!')

                        is_flipped = sample['meta']['flip']

                        
                        imgname = sample['meta']['current_name']
                        ori_height = sample['meta']['height']
                        ori_width = sample['meta']['width']
                        obj_idx = sample['meta']['obj_idx']

                        if not self.cfg.TEST_PANO:
                            obj_nums = sample['meta']['obj_num']
                            obj_nums = [int(obj_num) for obj_num in obj_nums]
                        obj_idx = [int(_obj_idx) for _obj_idx in obj_idx]

                        current_img = sample['current_img']
                        current_img = current_img.cuda(self.gpu,
                                                       non_blocking=True)
                        sample['current_img'] = current_img

                        if 'current_label' in sample.keys():
                            current_label = sample['current_label'].cuda(
                                self.gpu, non_blocking=True).float()
                            if self.cfg.TEST_PANO:
                                obj_nums = sample['meta']['obj_num']
                                obj_nums = [int(_obj_nums) for _obj_nums in obj_nums]
                                obj_mapping = sample['meta']['obj_mapping']
                        else:
                            current_label = None

                        #############################################################

                        if aug_idx == 0:
                            start.record()

                        if frame_idx == 0:
                            if cfg.TEST_BOX_FILTER:
                                box_dict.update(label2box(current_label.squeeze(0)))
                                box_dict = check_box(box_dict,current_img.shape[2],current_img.shape[3])
                            _current_label = F.interpolate(
                                current_label,
                                size=current_img.size()[2:],
                                mode="nearest")
                            engine.add_reference_frame(current_img,
                                                       _current_label,
                                                       frame_step=0,
                                                       obj_nums=obj_mapping if self.cfg.TEST_PANO else obj_nums)
                        else:
                            engine.match_propogate_one_frame(current_img)
                            # downsampled pred_logit
                            pred_logit = engine.decode_current_logits()

                            if is_flipped:
                                pred_logit = flip_tensor(pred_logit, 3)

                            pred_logit_resized = F.interpolate(pred_logit,size=(ori_height,ori_width),
                                                    mode='bilinear',align_corners=cfg.MODEL_ALIGN_CORNERS)
                            pred_prob = torch.softmax(pred_logit_resized, dim=1)
                            
                            all_preds.append(pred_prob)

                            is_inplace_flipped = cfg.TEST_INPLACE_FLIP
                            if not is_flipped and current_label is not None and new_obj_label is None:
                                new_obj_label = current_label
                            if is_inplace_flipped  and current_label is not None and new_obj_label is None:
                                new_obj_label = flip_tensor(current_label,3)

                    if frame_idx > 0:
                        all_pred_probs = [
                            torch.mean(pred, dim=0, keepdim=True)
                            for pred in all_preds
                        ]
                        all_pred_labels = [
                            torch.argmax(prob, dim=1, keepdim=True).float()
                            for prob in all_pred_probs
                        ]

                        cat_all_preds = torch.cat(all_preds, dim=0)
                        pred_prob = torch.mean(cat_all_preds,
                                               dim=0,
                                               keepdim=True)
                        pred_label = torch.argmax(pred_prob,
                                                  dim=1,
                                                  keepdim=True).float()
                        if cfg.TEST_BOX_FILTER:
                            new_box_dict = label2box(pred_label.squeeze(0))
                            pred_label,pred_prob = box_filter(box_dict,pred_label,pred_prob)
                            for label,prob in zip(all_pred_labels,all_pred_probs):
                                label,prob = box_filter(box_dict,label,prob)
                            box_dict.update(label2box(pred_label.squeeze(0)))
                            box_dict = check_box(box_dict,pred_label.shape[2],pred_label.shape[3])

                        if new_obj_label is not None:
                            keep = (new_obj_label == 0).float()
                            all_pred_labels = [label * \
                                keep + new_obj_label * (1 - keep) for label in all_pred_labels]

                            pred_label = pred_label * \
                                keep + new_obj_label * (1 - keep)
                            new_obj_nums = [int(pred_label.max().item())]
                            if cfg.TEST_BOX_FILTER:
                                box_dict.update(label2box(pred_label.squeeze(0)))
                                box_dict = check_box(box_dict,pred_label.shape[2],pred_label.shape[3])
                            
                            if cfg.TEST_SAVE_PROB:
                                # !!! bug exits when obj>20
                                # update pred_prob by new obj label
                                for idx in range(obj_nums[0]+1,new_obj_nums[0]+1):
                                    tmp_mask = new_obj_label.repeat(1,pred_prob.shape[1],1,1)==idx
                                    pred_prob[tmp_mask] = 0.
                                    new_prob_layer = torch.zeros_like(pred_prob[:,0])
                                    new_prob_layer[new_obj_label.squeeze(0)==idx] = 1.
                                    pred_prob = torch.cat([pred_prob,new_prob_layer.unsqueeze(0)],axis=1)
                                    
                                    # new_obj_label_s = F.interpolate(new_obj_label,size=pred_logit.shape[2:],mode='nearest')
                                    # tmp_mask = new_obj_label_s.repeat(1,pred_prob.shape[1],1,1)==idx
                                    # shift_val = 1e+10 if pred_logit.dtype==torch.float32 else 1e+4
                                    # pred_logit[tmp_mask] = -shift_val
                                    # pred_logit[:,idx][new_obj_label_s.squeeze(0)==idx] = shift_val

                            if cfg.TEST_FLIP:
                                all_flip_pred_labels = [
                                    flip_tensor(label, 3)
                                    for label in all_pred_labels
                                ]
                                flip_pred_label = flip_tensor(pred_label, 3)

                            for aug_idx in range(len(samples)):
                                engine = all_engines[aug_idx]
                                current_img = samples[aug_idx]['current_img']

                                # current_label = flip_pred_label if samples[
                                #     aug_idx]['meta']['flip'] else pred_label
                                current_label = all_flip_pred_labels[
                                    aug_idx] if samples[aug_idx]['meta'][
                                        'flip'] else all_pred_labels[aug_idx]
                                current_label = F.interpolate(
                                    current_label,
                                    size=engine.input_size_2d,
                                    mode="nearest")
                                engine.add_reference_frame(
                                    current_img,
                                    current_label,
                                    obj_nums=obj_mapping if self.cfg.TEST_PANO else new_obj_nums,
                                    frame_step=frame_idx)
                        else:
                            if not cfg.MODEL_USE_PREV_PROB:
                                if cfg.TEST_FLIP:
                                    all_flip_pred_labels = [
                                        flip_tensor(label, 3)
                                        for label in all_pred_labels
                                    ]
                                    flip_pred_label = flip_tensor(
                                        pred_label, 3)

                                for aug_idx in range(len(samples)):
                                    engine = all_engines[aug_idx]
                                    # current_label = flip_pred_label if samples[
                                    #     aug_idx]['meta']['flip'] else pred_label
                                    current_label = all_flip_pred_labels[
                                        aug_idx] if samples[aug_idx]['meta'][
                                            'flip'] else all_pred_labels[
                                                aug_idx]
                                    current_label = F.interpolate(
                                        current_label,
                                        size=engine.input_size_2d,
                                        mode="nearest")
                                    engine.update_memory(current_label)
                            else:
                                if cfg.TEST_FLIP:
                                    all_flip_pred_probs = [
                                        flip_tensor(prob, 3)
                                        for prob in all_pred_probs
                                    ]
                                    flip_pred_prob = flip_tensor(pred_prob, 3)

                                for aug_idx in range(len(samples)):
                                    engine = all_engines[aug_idx]
                                    # current_prob = flip_pred_prob if samples[
                                    #     aug_idx]['meta']['flip'] else pred_prob
                                    current_label = all_flip_pred_probs[
                                        aug_idx] if samples[aug_idx]['meta'][
                                            'flip'] else all_pred_probs[aug_idx]
                                    current_prob = F.interpolate(
                                        current_prob,
                                        size=engine.input_size_2d,
                                        mode="nearest")
                                    engine.update_memory(current_prob)

                        end.record()
                        torch.cuda.synchronize()
                        one_frametime += start.elapsed_time(end) / 1e3

                        seq_total_time += one_frametime
                        seq_total_frame += 1
                        if cfg.TEST_PANO:
                            obj_num = obj_nums[0] + obj_nums[1]
                        else:
                            obj_num = obj_nums[0]
                        if cfg.TEST_FRAME_LOG:
                            print(
                                'GPU {} - Frame: {} - Obj Num: {}, Time: {}ms'.
                                format(self.gpu, imgname[0].split('.')[0],
                                       obj_num, int(one_frametime * 1e3)))
                        # Save result
                        save_mask(
                            pred_label.squeeze(0).squeeze(0),
                            os.path.join(self.result_root, seq_name,
                                         imgname[0].split('.')[0] + '.png'),
                            obj_idx)
                        if 'all_frames' in cfg.TEST_DATASET_SPLIT and imgname in images_sparse:
                            save_mask(
                                pred_label.squeeze(0).squeeze(0),
                                os.path.join(self.result_root_sparse, seq_name,
                                             imgname[0].split('.')[0] +
                                             '.png'), obj_idx)
                        if cfg.TEST_SAVE_PROB:
                            if 'all_frames' not in cfg.TEST_DATASET_SPLIT:
                                save_prob(pred_prob, #(B,C,H,W)
                                    os.path.join(self.result_root_prob,seq_name,imgname[0].split('.')[0] +
                                                '.npy'), obj_idx,
                                                scale=cfg.TEST_SAVE_PROB_SCALE)
                            elif imgname in images_sparse:
                                save_prob(pred_prob,#(B,C,H,W)
                                    os.path.join(self.result_root_prob,seq_name,imgname[0].split('.')[0] +
                                                '.npy'), obj_idx,
                                                scale=cfg.TEST_SAVE_PROB_SCALE)
                        if cfg.TEST_SAVE_LOGIT:
                            if 'all_frames' not in cfg.TEST_DATASET_SPLIT:
                                save_logit(pred_logit.squeeze(0), #(C,H,W)
                                    os.path.join(self.result_root_logit,seq_name,imgname[0].split('.')[0] +
                                                '.pt'), obj_idx)
                            elif imgname in images_sparse:
                                save_logit(pred_logit.squeeze(0),#(C,H,W)
                                    os.path.join(self.result_root_logit,seq_name,imgname[0].split('.')[0] +
                                                '.pt'), obj_idx)
                seq_avg_time_per_frame = seq_total_time / seq_total_frame
                total_time += seq_total_time
                total_frame += seq_total_frame
                total_avg_time_per_frame = total_time / total_frame
                total_sfps += seq_avg_time_per_frame
                avg_sfps = total_sfps / processed_video_num
                max_mem = torch.cuda.max_memory_allocated(
                    device=self.gpu) / (1024.**3)
                print(
                    "GPU {} - Seq {} - FPS: {:.2f}. All-Frame FPS: {:.2f}, All-Seq FPS: {:.2f}, Max Mem: {:.2f}G"
                    .format(self.gpu, seq_name, 1. / seq_avg_time_per_frame,
                            1. / total_avg_time_per_frame, 1. / avg_sfps,
                            max_mem))

        if self.seq_queue is not None:
            if self.rank != 0:
                self.info_queue.put({
                    'total_time': total_time,
                    'total_frame': total_frame,
                    'total_sfps': total_sfps,
                    'processed_video_num': processed_video_num,
                    'max_mem': max_mem
                })
            print('Finished the evaluation on GPU {}.'.format(self.gpu))
            if self.rank == 0:
                for _ in range(self.gpu_num - 1):
                    info_dict = self.info_queue.get()
                    total_time += info_dict['total_time']
                    total_frame += info_dict['total_frame']
                    total_sfps += info_dict['total_sfps']
                    processed_video_num += info_dict['processed_video_num']
                    max_mem = max(max_mem, info_dict['max_mem'])
                all_reduced_total_avg_time_per_frame = total_time / total_frame
                all_reduced_avg_sfps = total_sfps / processed_video_num
                print(
                    "GPU {} - All-Frame FPS: {:.2f}, All-Seq FPS: {:.2f}, Max Mem: {:.2f}G"
                    .format(list(range(self.gpu_num)),
                            1. / all_reduced_total_avg_time_per_frame,
                            1. / all_reduced_avg_sfps, max_mem))
        else:
            print(
                "GPU {} - All-Frame FPS: {:.2f}, All-Seq FPS: {:.2f}, Max Mem: {:.2f}G"
                .format(self.gpu, 1. / total_avg_time_per_frame, 1. / avg_sfps,
                        max_mem))

        if self.rank == 0:
            zip_folder(self.source_folder, self.zip_dir)
            self.print_log('Saving result to {}.'.format(self.zip_dir))
            if 'all_frames' in cfg.TEST_DATASET_SPLIT:
                zip_folder(self.result_root_sparse, self.zip_dir_sparse)
            end_eval_time = time.time()
            total_eval_time = str(
                datetime.timedelta(seconds=int(end_eval_time -
                                               start_eval_time)))
            self.print_log("Total evaluation time: {}".format(total_eval_time))

    def print_log(self, string):
        if self.rank == 0:
            print(string)