import importlib
import os
import sys
import random
import argparse

sys.path.append('.')
sys.path.append('..')

import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def setup_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main_worker(local_rank, cfg, args):
    world_size = cfg.TRAIN_GPUS

    use_cuda = torch.cuda.is_available() and world_size > 0

    # ---- Device handling ----
    if use_cuda:
        device_id = cfg.DIST_START_GPU + local_rank
        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')

    # ---- Init process group (ONLY for multi-GPU CUDA) ----
    if use_cuda and world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method=cfg.DIST_URL,
            world_size=world_size,
            rank=local_rank
        )

    # ---- Trainer ----
    from networks.managers.pre_trainer import StaticTrainer
    trainer = StaticTrainer(
        cfg,
        device=device,
        local_rank=local_rank,
        world_size=world_size if use_cuda else 1,
        use_amp=args.amp and use_cuda
    )

    trainer.train()

    if use_cuda and world_size > 1:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Static Pretraining (DDP Ready)")

    # ---- Experiment ----
    parser.add_argument('--exp_name', type=str, default='static_pretrain')
    parser.add_argument('--stage', type=str, default='pre')
    parser.add_argument('--model', type=str, default='aott')

    # ---- GPU ----
    parser.add_argument('--start_gpu', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=-1)

    # ---- Training ----
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=-1.0)
    parser.add_argument('--total_step', type=int, default=-1)

    # ---- AMP ----
    parser.add_argument('--amp', action='store_true')

    args = parser.parse_args()

    setup_seed()

    # ---- Load config ----
    engine_config = importlib.import_module('configs.' + args.stage)
    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    # ---- Apply overrides ----
    cfg.DIST_START_GPU = args.start_gpu
    cfg.TRAIN_GPUS = args.gpu_num


    if args.batch_size > 0:
        cfg.TRAIN_BATCH_SIZE = args.batch_size

    if args.lr > 0:
        cfg.TRAIN_LR = args.lr

    if args.total_step > 0:
        cfg.TRAIN_TOTAL_STEPS = args.total_step

    # ---- DDP URL ----
    cfg.DIST_URL = f"tcp://127.0.0.1:{random.randint(12000, 20000)}"

    print(f"[INFO] Starting training...")

    # ---- Launch ----
    if cfg.TRAIN_GPUS > 1:
        mp.spawn(
            main_worker,
            nprocs=cfg.TRAIN_GPUS,
            args=(cfg, args)
        )
    else:
        main_worker(0, cfg, args)


if __name__ == '__main__':
    main()