import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from dataloaders.train_datasets import StaticTrain
import dataloaders.video_transforms as tr

from networks.models import build_vos_model
from networks.engines import build_engine

from utils.learning import get_trainable_params, adjust_learning_rate
import torch.optim as optim

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
save_dir = os.path.join(BASE_DIR, "pretrain_models")

class StaticTrainer:
    """
    Clean isolated trainer for STATIC (image-based) pretraining.
    No temporal logic. No prev-frame dependency.
    """
    def __init__(self, cfg, gpu=0):
        self.cfg = cfg
        self.gpu = gpu

        torch.cuda.set_device(self.gpu)

        # ---- Build model ----
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(self.gpu)

        self.engine = build_engine(
            cfg.MODEL_ENGINE,
            'train',
            aot_model=self.model,
            gpu_id=self.gpu,
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

        self.loader = DataLoader(
            dataset,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATA_WORKERS,
            pin_memory=True
        )

    def train(self):
        self.model.train()
        cfg = self.cfg

        print(f"[StaticTrainer] Saving checkpoints to: {save_dir}")

        while self.step < cfg.TRAIN_TOTAL_STEPS:
            for sample in self.loader:
                if self.step >= cfg.TRAIN_TOTAL_STEPS:
                    break

                ref_imgs = sample['ref_img'].cuda(self.gpu)
                labels = sample['ref_label'].cuda(self.gpu)
                obj_nums = [int(x) for x in sample['meta']['obj_num']]

                batch_size = ref_imgs.size(0)

                # ---- Static mode ----
                all_frames = ref_imgs
                all_labels = labels

                self.engine.restart_engine(batch_size, True)
                self.optimizer.zero_grad()

                loss, _, _, _ = self.engine(
                    all_frames,
                    all_labels,
                    batch_size,
                    use_prev_pred=False,
                    obj_nums=obj_nums,
                    step=self.step,
                    tf_board=False,
                    enable_prev_frame=False,
                    use_prev_prob=0
                )

                loss = torch.mean(loss)

                loss.backward()
                self.optimizer.step()

                # ---- Logging ----
                if self.step % 50 == 0:
                    print(f"[Static Pretrain] Step {self.step} | Loss: {loss.item():.4f}")

                # ---- Checkpointing (SAFE & CONTROLLED) ----
                if self.step > 0 and self.step % 1000 == 0:
                    ckpt_path = os.path.join(save_dir, f"static_step_{self.step}.pth")
                    torch.save(self.model.state_dict(), ckpt_path)
                    print(f"[Checkpoint Saved] {ckpt_path}")

                self.step += 1

        # ---- Final checkpoint ----
        final_path = os.path.join(save_dir, "static_pretrain_final.pth")
        torch.save(self.model.state_dict(), final_path)

        print(f"[StaticTrainer] Final model saved at: {final_path}")
        print("Static pretraining finished.")    
