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
from common.utils import update_config
from common.config import _C as cfg

cfg = update_config(cfg, 'experiments/MM-Net_full.yaml')

dataset = ModulationsDataset(cfg.BASE_PATH, cfg.MODS, cfg.SNRS)

train_idx, val_idx = train_test_split(np.arange(len(dataset)),
                                      test_size=cfg.VAL_RATIO,
                                      random_state=123,
                                      shuffle=True,
                                      stratify=dataset.labels)

# Subset dataset for train and val
train_dataset = Subset(dataset, train_idx)
validation_dataset = Subset(dataset, val_idx)
STEPS = len(train_dataset) // cfg.BATCH_SIZE
VAL_STEPS = len(validation_dataset) // cfg.BATCH_SIZE
train_loader = DataLoader(train_dataset,
                          batch_size=cfg.BATCH_SIZE,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)

val_loader = DataLoader(validation_dataset,
                        batch_size=cfg.BATCH_SIZE,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

model = MMNet()
model.cuda()

optimizer = SGD(model.parameters(), cfg.LEARNING_RATE, momentum=0.9)

scheduler = StepLR(optimizer, step_size=2, gamma=.7)

epoch_losses = []
epoch_val_losses = []
def train():
    for epoch in range(cfg.NUM_EPOCHS):
        pbar = tqdm(train_loader, desc='Training Epoch {}/{}'.format(epoch, cfg.NUM_EPOCHS), unit='steps')
        model.train()
        cum_loss = 0
        for step, inp in enumerate(pbar):

            optimizer.zero_grad()

            images, cumulants, _, labels = inp

            images = images.cuda()
            cumulants = cumulants.cuda().float()
            labels = labels.cuda()

            logits = model(cumulants, images)

            batch_loss = CrossEntropyLoss()(logits, labels)
            batch_loss.backward()
            optimizer.step()
            cum_loss += batch_loss.item()

            pbar.set_postfix({"last_loss": batch_loss.item()})

        epoch_loss = cum_loss / STEPS
        epoch_losses.append(epoch_loss)
        scheduler.step()

        pbar = tqdm(val_loader, desc='Validating Epoch {}/{}'.format(epoch, cfg.NUM_EPOCHS), unit='steps')
        model.eval()
        cum_loss = 0
        for step, inp in enumerate(pbar):
            with torch.no_grad():
                images, cumulants, _, labels = inp

                images = images.cuda()
                cumulants = cumulants.cuda().float()
                labels = labels.cuda()

                logits = model(cumulants, images)

            batch_loss = CrossEntropyLoss()(logits, labels)

            cum_loss += batch_loss.item()
            pbar.set_postfix({"last_loss": batch_loss.item()})

        epoch_val_loss = cum_loss / VAL_STEPS
        epoch_val_losses.append(epoch_val_loss)
        if epoch_val_loss <= min(epoch_val_losses):
            torch.save(model.state_dict(), "./ckpt/" + cfg.MODEL_NAME)

if __name__ == "__main__":
    train()