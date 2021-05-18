"""
Training script to train DistinctNet on segmenting manipulator and a grasped object.
"""

import os
from time import strftime, gmtime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from networks.distinctnet import DistinctNet
from utils.train_loop import Trainer
from utils.train_utils import get_default_argparse
from data.semantic_dataset import SemanticDataset
from utils.confmat import ConfusionMatrixSemantic


def main():
    cfg = get_default_argparse()
    print(f"Running training with config\n{cfg}")

    # create output dirs, dump config and create writer
    os.makedirs(cfg.EXP.get("ROOT", "./runs"), exist_ok=True)
    output_path = os.path.join(cfg.EXP.get("ROOT", "./runs"), cfg.EXP.get("NAME", "semantic") + "_" + strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'tb'), exist_ok=True)
    with open(os.path.join(output_path, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    train_writer = SummaryWriter(log_dir=os.path.join(output_path, 'tb'))
    val_writer = SummaryWriter(log_dir=os.path.join(output_path, 'tb'))
    print(f"Saving outputs to {output_path}")

    print("Loading data ...")
    train_data = SemanticDataset(cfg=cfg.DATA, split='train')
    val_data = SemanticDataset(cfg=cfg.DATA, split='val')

    train_loader = DataLoader(train_data, batch_size=cfg.DATA.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.DATA.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=cfg.DATA.VAL.BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS, drop_last=True)
    print(f"Loaded {len(train_data)} training samples.")
    print(f"Loaded {len(val_data)} validation samples.")

    # networks
    print(f"Initializing DistinctNet ...")
    net = DistinctNet(cfg.MODEL).cuda()
    # load pretrained motion weights
    assert os.path.isfile(cfg.MODEL.WEIGHTS), f"Invalid weight path: {cfg.MODEL.WEIGHTS}"
    state_dict = torch.load(cfg.MODEL.WEIGHTS)
    # delete last layer as the shape will be different
    del state_dict['decoder.last_conv.8.weight']
    del state_dict['decoder.last_conv.8.bias']
    rets = net.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {cfg.MODEL.WEIGHTS}: {rets}")

    # loss, metric
    weight = torch.tensor([[[1.]], [[2.]], [[3.]], [[10.]]]).cuda() if cfg.MODEL.OUT_CH == 4 else torch.tensor([[[1.]], [[2.]], [[3.]], [[10.]], [[8.]]]).cuda()
    loss_fn = BCEWithLogitsLoss(weight=weight)
    metric_fn = ConfusionMatrixSemantic(num_classes=cfg.MODEL.OUT_CH, labels=['background', 'foreground', 'manipulator', 'object'] if cfg.MODEL.OUT_CH == 4 else ['background', 'foreground', 'manipulator', 'object', 'distractor'])

    # optimizer
    param_list = [
        {"params": net.encoder.parameters(), "lr": 0.0001},
        {"params": net.corr_layer.parameters(), "lr": 0.0001},
        {"params": net.reduction_layer.parameters(), "lr": 0.0001},
        {"params": net.aspp.parameters(), "lr": 0.0001},
        {"params": net.decoder.parameters(), "lr": 0.0001},
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
