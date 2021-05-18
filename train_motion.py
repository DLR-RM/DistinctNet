"""
Training script to train DistinctNet on segmenting agnostic moving objects.
"""

import os
from time import strftime, gmtime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from utils.train_utils import get_default_argparse
from utils.train_loop import Trainer
from utils.confmat import ConfusionMatrix
from networks.distinctnet import DistinctNet
from data.motion_dataset import MotionDataset


def main():
    cfg = get_default_argparse()
    print(f"Running training with config\n{cfg}")

    # create output dirs, dump config and create writer
    os.makedirs(cfg.EXP.get("ROOT", "./runs"), exist_ok=True)
    output_path = os.path.join(cfg.EXP.get("ROOT", "./runs"), cfg.EXP.get("NAME", "motion") + "_" + strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'tb'), exist_ok=True)
    with open(os.path.join(output_path, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    train_writer = SummaryWriter(log_dir=os.path.join(output_path, 'tb'))
    val_writer = SummaryWriter(log_dir=os.path.join(output_path, 'tb'))
    print(f"Saving outputs to {output_path}")

    print("Loading data ...")
    train_data = MotionDataset(cfg=cfg.DATA, split='train')
    val_data = MotionDataset(cfg=cfg.DATA, split='val')

    train_loader = DataLoader(train_data, batch_size=cfg.DATA.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.DATA.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=cfg.DATA.VAL.BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS, drop_last=True)
    print(f"Loaded {len(train_data)} training samples.")
    print(f"Loaded {len(val_data)} validation samples.")

    # networks
    print(f"Initializing DistinctNet ...")
    net = DistinctNet(cfg.MODEL).cuda()

    # loss, metric
    loss_fn = CrossEntropyLoss()
    metric_fn = ConfusionMatrix(num_classes=2, labels=['background', 'foreground'])

    # optimizer
    param_list = [
        {"params": net.intermediate.parameters(), "lr": 0.0001},
        {"params": net.siamese.parameters(), "lr": 0.0001},
        {"params": net.encoder.parameters(), "lr": 0.0001},
        {"params": net.corr_layer.parameters(), "lr": 0.0001},
        {"params": net.reduction_layer.parameters(), "lr": 0.0001},
        {"params": net.aspp.parameters(), "lr": 0.001},
        {"params": net.decoder.parameters(), "lr": 0.001},
    ]
    optimizer = torch.optim.AdamW(param_list, lr=0.001, weight_decay=0.01)

    # train loop
    trainer = Trainer(net=net, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, loss_fn=loss_fn, metric_fn=metric_fn, train_writer=train_writer, val_writer=val_writer, device=torch.device("cuda"))

    for ep in range(50):
        trainer(curr_epoch=ep)

        # save current model
        torch.save(trainer.net.state_dict(), os.path.join(output_path, f"models/model_{str(ep).zfill(2)}.pth"))


if __name__ == '__main__':
    main()
