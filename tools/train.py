import importlib
import random
import sys
import torch
import torch.multiprocessing as mp

sys.setrecursionlimit(10000)
sys.path.append('.')
sys.path.append('..')

from networks.managers.trainer import Trainer


def main_worker(rank, cfg, enable_amp=True):
    # -------- Device Resolution --------
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
        enable_amp = False  # AMP not supported on CPU

    # -------- Debug Info (important) --------
    print(f"[INFO] Rank: {rank} | Device: {device} | AMP: {enable_amp}")

    # -------- Trainer Init --------
    trainer = Trainer(
        rank=rank,
        cfg=cfg,
        enable_amp=enable_amp,
        device=device  # <-- enforce device injection
    )

    trainer.sequential_training()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train VOS")

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--stage', type=str, default='pre')
    parser.add_argument('--model', type=str, default='aott')
    parser.add_argument('--max_id_num', type=int, default='-1')

    parser.add_argument('--start_gpu', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--dist_url', type=str, default='')
    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)

    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--datasets', nargs='+', type=str, default=[])
    parser.add_argument('--lr', type=float, default=-1.)
    parser.add_argument('--total_step', type=int, default=-1.)
    parser.add_argument('--start_step', type=int, default=-1.)

    args = parser.parse_args()

    # -------- Load Config --------
    engine_config = importlib.import_module('configs.' + args.stage)
    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    # -------- Override Config --------
    if len(args.datasets) > 0:
        cfg.DATASETS = args.datasets

    cfg.DIST_START_GPU = args.start_gpu

    if args.gpu_num > 0:
        cfg.TRAIN_GPUS = args.gpu_num

    if args.batch_size > 0:
        cfg.TRAIN_BATCH_SIZE = args.batch_size

    if args.pretrained_path != '':
        cfg.PRETRAIN_MODEL = args.pretrained_path

    if args.max_id_num > 0:
        cfg.MODEL_MAX_OBJ_NUM = args.max_id_num

    if args.lr > 0:
        cfg.TRAIN_LR = args.lr

    if args.total_step > 0:
        cfg.TRAIN_TOTAL_STEPS = args.total_step

    if args.start_step > 0:
        cfg.TRAIN_START_STEP = args.start_step

    # -------- Dist URL --------
    if args.dist_url == '':
        cfg.DIST_URL = 'tcp://127.0.0.1:123' + str(random.randint(0, 99))
    else:
        cfg.DIST_URL = args.dist_url

    # -------- Device Mode Decision --------
    use_cuda = torch.cuda.is_available()

    if not use_cuda:
        print("[INFO] CUDA not available → Running on CPU")
        cfg.TRAIN_GPUS = 0
        main_worker(0, cfg, enable_amp=False)

    elif cfg.TRAIN_GPUS > 1:
        print(f"[INFO] Multi-GPU Training on {cfg.TRAIN_GPUS} GPUs")
        mp.spawn(main_worker, nprocs=cfg.TRAIN_GPUS, args=(cfg, args.amp))

    else:
        print("[INFO] Single GPU Training")
        cfg.TRAIN_GPUS = 1
        main_worker(0, cfg, enable_amp=args.amp)


if __name__ == '__main__':
    main()
    