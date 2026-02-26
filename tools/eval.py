import importlib
import sys
import torch
import torch.multiprocessing as mp

sys.path.append('.')
sys.path.append('..')

from networks.managers.evaluator import Evaluator


# ─────────────────────────────────────────────
# Device utilities
# ─────────────────────────────────────────────
def get_device(gpu_id=0):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[Device] CUDA:{gpu_id} — {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (CUDA not available)")
    return device


def use_amp(enable_flag, device):
    """Return context manager for AMP if valid."""
    if enable_flag and device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    else:
        # dummy context
        from contextlib import nullcontext
        return nullcontext()


# ─────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────
def main_worker(rank, cfg, seq_queue=None, info_queue=None, enable_amp=False):
    device = get_device(rank if torch.cuda.is_available() else 0)

    evaluator = Evaluator(
        rank=rank if device.type == "cuda" else None,
        cfg=cfg,
        seq_queue=seq_queue,
        info_queue=info_queue
    )

    with use_amp(enable_amp, device):
        evaluator.evaluating()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Eval VOS")

    parser.add_argument('--exp_name', type=str, default='default')

    parser.add_argument('--stage', type=str, default='pre')
    parser.add_argument('--model', type=str, default='aott')
    parser.add_argument('--lstt_num', type=int, default=-1)
    parser.add_argument('--lt_gap', type=int, default=-1)
    parser.add_argument('--st_skip', type=int, default=-1)
    parser.add_argument('--max_id_num', type=int, default='-1')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help="GPU id (ignored if CUDA unavailable)")
    parser.add_argument('--gpu_num', type=int, default=1,
                        help="Number of GPUs (ignored if CUDA unavailable)")

    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--ckpt_step', type=int, default=-1)

    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--split', type=str, default='')

    parser.add_argument('--ema', action='store_true')
    parser.set_defaults(ema=False)

    parser.add_argument('--flip', action='store_true')
    parser.set_defaults(flip=False)
    parser.add_argument('--ms', nargs='+', type=float, default=[1.])

    parser.add_argument('--max_resolution', type=float, default=480 * 1.3)

    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)

    args = parser.parse_args()

    # ── Config load ─────────────────────────────
    engine_config = importlib.import_module('configs.' + args.stage)
    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    cfg.TEST_EMA = args.ema

    cfg.TEST_GPU_ID = args.gpu_id
    cfg.TEST_GPU_NUM = args.gpu_num if torch.cuda.is_available() else 0

    if args.lstt_num > 0:
        cfg.MODEL_LSTT_NUM = args.lstt_num
    if args.lt_gap > 0:
        cfg.TEST_LONG_TERM_MEM_GAP = args.lt_gap
    if args.st_skip > 0:
        cfg.TEST_SHORT_TERM_MEM_SKIP = args.st_skip
    if args.max_id_num > 0:
        cfg.MODEL_MAX_OBJ_NUM = args.max_id_num

    if args.ckpt_path != '':
        cfg.TEST_CKPT_PATH = args.ckpt_path
    if args.ckpt_step > 0:
        cfg.TEST_CKPT_STEP = args.ckpt_step

    if args.dataset != '':
        cfg.TEST_DATASET = args.dataset
    if args.split != '':
        cfg.TEST_DATASET_SPLIT = args.split

    cfg.TEST_FLIP = args.flip
    cfg.TEST_MULTISCALE = args.ms

    if cfg.TEST_MULTISCALE != [1.]:
        cfg.TEST_MAX_SHORT_EDGE = args.max_resolution
    else:
        cfg.TEST_MAX_SHORT_EDGE = None

    cfg.TEST_MAX_LONG_EDGE = args.max_resolution * 800. / 480.

    # ── Launch logic ────────────────────────────
    if torch.cuda.is_available() and args.gpu_num > 1:
        print(f"[Launch] Multi-GPU mode ({args.gpu_num} GPUs)")
        mp.set_start_method('spawn', force=True)

        seq_queue = mp.Queue()
        info_queue = mp.Queue()

        mp.spawn(
            main_worker,
            nprocs=args.gpu_num,
            args=(cfg, seq_queue, info_queue, args.amp)
        )

    else:
        if args.gpu_num > 1 and not torch.cuda.is_available():
            print("[Warning] gpu_num > 1 ignored because CUDA not available.")

        print("[Launch] Single device mode")
        main_worker(args.gpu_id, cfg, enable_amp=args.amp)


# ─────────────────────────────────────────────
if __name__ == '__main__':
    main()
    