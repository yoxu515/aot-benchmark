import importlib
import os
import sys
import random
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.managers.trainer import Trainer
sys.path.append('.')
sys.path.append('..')

# ----------------------------
# Argument Parser
# ----------------------------
def parse_args():
    import argparse
    parser = argparse.ArgumentParser("Train VOS")

    # experiment
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--stage', type=str, default='pre')
    parser.add_argument('--model', type=str, default='aott')

    # training overrides
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=-1.)
    parser.add_argument('--total_step', type=int, default=-1)

    # data / pretrained
    parser.add_argument('--datasets', nargs='+', type=str, default=[])
    parser.add_argument('--pretrained_path', type=str, default='')

    # AMP
    parser.add_argument('--amp', action='store_true')

    return parser.parse_args()


# ----------------------------
# Runtime Setup
# ----------------------------
def setup_runtime():
    use_cuda = torch.cuda.is_available()
    # CPU fallback
    if not use_cuda:
        return {
            "device": torch.device("cpu"),
            "rank": 0,
            "world_size": 1,
            "distributed": False
        }

    # DDP via torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        device = torch.device("cuda", local_rank)

        return {
            "device": device,
            "rank": rank,
            "world_size": world_size,
            "distributed": True
        }

    # Single GPU fallback
    torch.cuda.set_device(0)
    return {
        "device": torch.device("cuda:0"),
        "rank": 0,
        "world_size": 1,
        "distributed": False
    }


# ----------------------------
# Seeding (per-rank)
# ----------------------------
def set_seed(base_seed, rank):
    seed = base_seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    # -------- Load config --------
    engine_config = importlib.import_module('configs.' + args.stage)
    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    # -------- Override config --------
    if args.datasets:
        cfg.DATASETS = args.datasets
    if args.batch_size > 0:
        cfg.TRAIN_BATCH_SIZE = args.batch_size
    if args.lr > 0:
        cfg.TRAIN_LR = args.lr
    if args.total_step > 0:
        cfg.TRAIN_TOTAL_STEPS = args.total_step
    if args.pretrained_path:
        cfg.PRETRAIN_MODEL = args.pretrained_path

    # -------- Runtime --------
    runtime = setup_runtime()

    device = runtime["device"]
    rank = runtime["rank"]
    world_size = runtime["world_size"]
    distributed = runtime["distributed"]

    cfg.TRAIN_GPUS = world_size
    cfg.DIST_ENABLE = distributed

    # -------- Seeding --------
    set_seed(42, rank)

    # -------- Logging --------
    if rank == 0:
        print(f"[INFO] Rank {rank} | Device {device} | World Size {world_size}")

    # -------- AMP --------
    enable_amp = args.amp and device.type == "cuda"

    # -------- Trainer --------
    trainer = Trainer(
        rank=rank,
        world_size=world_size,
        cfg=cfg,
        enable_amp=enable_amp,
        device=device
    )

    # -------- Train --------
    trainer.sequential_training()

    # -------- Cleanup --------
    if distributed:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()