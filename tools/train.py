import importlib
import random
import sys
import torch
import torch.multiprocessing as mp
from contextlib import nullcontext

sys.setrecursionlimit(10000)
sys.path.append('.')
sys.path.append('..')

from networks.managers.trainer import Trainer


# ─────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────
def get_device(rank=0):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        print(f"[Device] CUDA:{rank} — {torch.cuda.get_device_name(rank)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU fallback")
    return device


def amp_context(enable_amp, device):
    if enable_amp and device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


# ─────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────
def main_worker(rank, cfg, enable_amp=True):

    device = get_device(rank if torch.cuda.is_available() else 0)

    trainer = Trainer(
        rank=rank if device.type == "cuda" else None,
        cfg=cfg,
        enable_amp=(enable_amp and device.type == "cuda")
    )

    with amp_context(enable_amp, device):
        trainer.sequential_training()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train VOS")

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--stage', type=str, default='pre')
    parser.add_argument('--model', type=str, default='aott')
    parser.add_argument('--max_id_num', type=int, default=-1)

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

    # ── Load config ───────────────────────────
    engine_config = importlib.import_module('configs.' + args.stage)
    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    # ── Apply overrides ───────────────────────
    if args.datasets:
        cfg.DATASETS = args.datasets

    cfg.DIST_START_GPU = args.start_gpu

    if args.gpu_num > 0:
        cfg.TRAIN_GPUS = args.gpu_num

    if args.batch_size > 0:
        cfg.TRAIN_BATCH_SIZE = args.batch_size

    if args.pretrained_path:
        cfg.PRETRAIN_MODEL = args.pretrained_path

    if args.max_id_num > 0:
        cfg.MODEL_MAX_OBJ_NUM = args.max_id_num

    if args.lr > 0:
        cfg.TRAIN_LR = args.lr

    if args.total_step > 0:
        cfg.TRAIN_TOTAL_STEPS = args.total_step

    if args.start_step > 0:
        cfg.TRAIN_START_STEP = args.start_step

    # ── Dist URL ──────────────────────────────
    if args.dist_url == '':
        cfg.DIST_URL = f"tcp://127.0.0.1:123{random.randint(0,9)}{random.randint(0,9)}"
    else:
        cfg.DIST_URL = args.dist_url

    # ── GPU availability handling ─────────────
    if not torch.cuda.is_available():
        print("[Warning] CUDA not available — forcing CPU mode.")
        cfg.TRAIN_GPUS = 0

    # ── Launch ────────────────────────────────
    if torch.cuda.is_available() and cfg.TRAIN_GPUS > 1:

        print(f"[Launch] Distributed training on {cfg.TRAIN_GPUS} GPUs")

        mp.set_start_method("spawn", force=True)

        mp.spawn(
            main_worker,
            nprocs=cfg.TRAIN_GPUS,
            args=(cfg, args.amp)
        )

    else:
        if cfg.TRAIN_GPUS > 1 and not torch.cuda.is_available():
            print("[Warning] Multi-GPU requested but CUDA unavailable → single CPU mode")

        cfg.TRAIN_GPUS = 1
        print("[Launch] Single device training")

        main_worker(args.start_gpu, cfg, args.amp)


# ─────────────────────────────────────────────
if __name__ == '__main__':
    main()
    