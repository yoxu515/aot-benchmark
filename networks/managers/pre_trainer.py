import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from dataloaders.train_datasets import StaticTrain
import dataloaders.video_transforms as tr

from networks.models import build_vos_model
from networks.engines import build_engine

from utils.learning import get_trainable_params
import torch.optim as optim


save_dir = os.path.join(BASE_DIR, "pretrain_models")

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


class StaticTrainer:
    def __init__(self, cfg, device, local_rank=0, world_size=1, use_amp=False):
        self.cfg = cfg
        self.device = torch.device(device)
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_amp = use_amp

        print(f"[StaticTrainer][Rank {local_rank}] Using device: {self.device}")

        # ---- Model ----
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).to(self.device)

        # ---- DDP wrap ----
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device.index])

        # ---- Engine ----
        self.engine = build_engine(
            cfg.MODEL_ENGINE,
            'train',
            aot_model=self.model,
            gpu_id=self.device.index if self.device.type == "cuda" else -1,
            long_term_mem_gap=cfg.TRAIN_LONG_TERM_MEM_GAP
        )

        # ---- Optimizer ----
        trainable_params = get_trainable_params(
            model=self.engine,
            base_lr=cfg.TRAIN_LR,
            weight_decay=cfg.TRAIN_WEIGHT_DECAY
        )

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=cfg.TRAIN_LR,
            weight_decay=cfg.TRAIN_WEIGHT_DECAY
        )

        # ---- AMP ----
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # ---- Dataset ----
        self.prepare_dataset()

        self.step = 0
    def prepare_dataset(self):
        cfg = self.cfg

        composed_transforms = transforms.Compose([
            tr.RandomScale(cfg.DATA_MIN_SCALE_FACTOR,
                        cfg.DATA_MAX_SCALE_FACTOR,
                        cfg.DATA_SHORT_EDGE_LEN),
            tr.BalancedRandomCrop(cfg.DATA_RANDOMCROP,
                                max_obj_num=cfg.MODEL_MAX_OBJ_NUM),
            tr.RandomHorizontalFlip(cfg.DATA_RANDOMFLIP),
            tr.Resize(cfg.DATA_RANDOMCROP, use_padding=True),
            tr.ToTensor()
        ])

        dataset = StaticTrain(
            cfg.DIR_STATIC,
            cfg.DATA_RANDOMCROP,
            seq_len=cfg.DATA_SEQ_LEN,
            merge_prob=cfg.DATA_DYNAMIC_MERGE_PROB,
            max_obj_n=cfg.MODEL_MAX_OBJ_NUM,
            aug_type=cfg.TRAIN_AUG_TYPE
        )

        # ---- DDP sampler ----
        if self.world_size > 1:
            self.sampler = DistributedSampler(dataset)
        else:
            self.sampler = None

        self.loader = DataLoader(
            dataset,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            sampler=self.sampler,
            shuffle=(self.sampler is None),
            num_workers=cfg.DATA_WORKERS,
            pin_memory=(self.device.type == "cuda")
        )

    def train(self):
        self.model.train()
        cfg = self.cfg

        if self.local_rank == 0:
            print(f"[StaticTrainer] Saving checkpoints to: {save_dir}")

        while self.step < cfg.TRAIN_TOTAL_STEPS:

            if self.sampler is not None:
                self.sampler.set_epoch(self.step)

            for sample in self.loader:
                print(f"Step {self.step}")
                if self.step >= cfg.TRAIN_TOTAL_STEPS:
                    break

                ref_imgs = sample['ref_img'].to(self.device, non_blocking=True)
                labels = sample['ref_label'].to(self.device, non_blocking=True)
                obj_nums = [int(x) for x in sample['meta']['obj_num']]

                batch_size = ref_imgs.size(0)

                self.engine.restart_engine(batch_size, True)
                self.optimizer.zero_grad()

                # ---- AMP ----
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss, _, _, _ = self.engine(
                        ref_imgs,
                        labels,
                        batch_size,
                        use_prev_pred=False,
                        obj_nums=obj_nums,
                        step=self.step,
                        tf_board=False,
                        enable_prev_frame=False,
                        use_prev_prob=0
                    )

                    loss = torch.mean(loss)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # ---- Logging (ONLY rank 0) ----
                if self.local_rank == 0 and self.step % 50 == 0:
                    print(f"[Static Pretrain] Step {self.step} | Loss: {loss.item():.4f}")

                # ---- Checkpoint (ONLY rank 0) ----
                if self.local_rank == 0 and self.step > 0 and self.step % 1000 == 0:
                    ckpt_path = os.path.join(save_dir, f"static_step_{self.step}.pth")
                    torch.save(self.model.state_dict(), ckpt_path)
                    print(f"[Checkpoint Saved] {ckpt_path}")

                self.step += 1

        # ---- Final save ----
        if self.local_rank == 0:
            final_path = os.path.join(save_dir, "static_pretrain_final.pth")
            torch.save(self.model.state_dict(), final_path)
            print(f"[StaticTrainer] Final model saved at: {final_path}")