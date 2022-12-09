"""
CRAFT TEXT DETECTER

Custom CRAFT model trainer for high resolution document image using
pytorch-lightning.

author: YongWook Ha @ NHN Diquest
"""

import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from models.craft import CRAFT
from datasets.craft_dataset import AIhub_Dataset, AIhub_collate
from torch.utils.data import DataLoader

from utils.base import load_setting

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, default="../settings/default.yaml",
                        help="Experiment settings")
    parser.add_argument("--version", "-v", type=int, default=1,
                        help="Train experiment version")
    parser.add_argument("--num_workers", "-nw", type=int, default=16,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", "-bs", type=int, default=16,
                        help="batch size")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # setting
    cfg = load_setting(args.setting)
    device_cnt = 0
    cfg.craft.gpus = device_cnt
    cfg.update(vars(args))
    print("setting:", cfg)

    model = CRAFT(cfg).to(torch.device(f'cuda:{device_cnt}'))
    saved = torch.load(cfg.craft.weight, map_location=torch.device(f'cuda:{device_cnt}'))
    model.load_state_dict(saved)

    # 다양한 한글 데이터
    custom_collate = AIhub_collate(cfg=cfg.craft)

    train_set = AIhub_Dataset(original_data_dir = cfg.original_data_dir,
                              data_list_file = cfg.train_data_list_file)

    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers,
                                collate_fn=custom_collate)

    val_set = AIhub_Dataset(original_data_dir=cfg.original_data_dir,
                            data_list_file = cfg.valid_data_list_file)

    valid_dataloader = DataLoader(val_set, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers,
                                collate_fn=custom_collate)

    logger = TensorBoardLogger("document/tb_logs", name="model", version=cfg.version,
                               default_hp_metric=False)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="fscore",
        dirpath=f"{cfg.model_path}/version_{cfg.version}",
        filename="checkpoints-{epoch:02d}-{fscore:.3f}",
        save_top_k=3,
        mode="max",
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(gpus=1, max_epochs=cfg.epochs,
                        logger=logger, num_sanity_val_steps=0,
                        strategy="dp" if device_cnt > 1 else None,
                        callbacks=[ckpt_callback, lr_callback],
                        resume_from_checkpoint=cfg.load_chkpt if cfg.load_chkpt else None,
                        precision=16)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
