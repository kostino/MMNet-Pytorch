import logging

import torch
from tqdm import tqdm

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

from models.mmnet import MMNet
from data.datasets import ModulationsDataset
from common.utils import update_config, AverageMeter
from common.config import _C as cfg

cfg = update_config(cfg, 'experiments/MM-Net-ENv2_s.yaml')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if 'cuda' in device:
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

dataset = ModulationsDataset(cfg.DATA.BASE_PATH, cfg.DATA.MODS, cfg.DATA.SNRS)

train_idx, val_idx = train_test_split(np.arange(len(dataset)),
                                      test_size=cfg.TRAIN.VAL_RATIO,
                                      random_state=123,
                                      shuffle=True,
                                      stratify=dataset.labels)

# Subset dataset for train and val
train_dataset = Subset(dataset, train_idx)
validation_dataset = Subset(dataset, val_idx)
STEPS = len(train_dataset) // cfg.TRAIN.BATCH_SIZE
VAL_STEPS = len(validation_dataset) // cfg.TRAIN.BATCH_SIZE
train_loader = DataLoader(train_dataset,
                          batch_size=cfg.TRAIN.BATCH_SIZE,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)

val_loader = DataLoader(validation_dataset,
                        batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

model = MMNet(backbone=cfg.MODEL.BACKBONE)
model.to(device)

optimizer = SGD(model.parameters(), cfg.TRAIN.LEARNING_RATE, momentum=0.9)

scheduler = StepLR(optimizer, step_size=2, gamma=.7)

epoch_losses = []
epoch_val_losses = []
def train():
    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        pbar = tqdm(train_loader, desc='Training Epoch {}/{}'.format(epoch, cfg.TRAIN.NUM_EPOCHS), unit='steps')
        model.train()
        epoch_loss = AverageMeter()
        for step, inp in enumerate(pbar):

            optimizer.zero_grad()

            images, cumulants, _, labels = inp

            images = images.to(device, non_blocking=True)
            cumulants = cumulants.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)

            logits = model(cumulants, images)

            batch_loss = CrossEntropyLoss()(logits, labels)
            batch_loss.backward()
            optimizer.step()
            epoch_loss.update(batch_loss.item())

            pbar.set_postfix({"last_loss": batch_loss.item(), "epoch_loss": epoch_loss.average()})

        epoch_losses.append(epoch_loss.average())
        scheduler.step()

        pbar = tqdm(val_loader, desc='Validating Epoch {}/{}'.format(epoch, cfg.TRAIN.NUM_EPOCHS), unit='steps')
        model.eval()
        epoch_val_loss = AverageMeter()
        for step, inp in enumerate(pbar):
            with torch.no_grad():
                images, cumulants, _, labels = inp

                images = images.to(device, non_blocking=True)
                cumulants = cumulants.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True)

                logits = model(cumulants, images)

            batch_loss = CrossEntropyLoss()(logits, labels)

            epoch_val_loss.update(batch_loss.item())
            pbar.set_postfix({"last_loss": batch_loss.item(), "epoch_loss": epoch_val_loss.average()})

        epoch_val_losses.append(epoch_val_loss.average())
        if epoch_val_loss.average() <= min(epoch_val_losses):
            torch.save(model.state_dict(), "./ckpt/" + cfg.MODEL.NAME)

if __name__ == "__main__":
    train()