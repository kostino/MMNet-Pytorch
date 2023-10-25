import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

from models.fifnet import FiFNet
from models.ccnn import CCNN
from data.datasets import ModulationsDataset
from common.utils import update_config
from common.config import _C as cfg

cfg = update_config(cfg, 'experiments/DrCNN.yaml')

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

VAL_STEPS = len(validation_dataset) // cfg.TRAIN.BATCH_SIZE

val_loader = DataLoader(validation_dataset,
                        batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

model = FiFNet(classes=8, in_channels=3, hocs=True)
model.load_state_dict(torch.load('./ckpt/FiFNet_hocs'))
model.to(device)


def test():
    scores = torch.tensor([]).to(device)
    lab = torch.tensor([]).to(device)
    snr = []
    pbar = tqdm(val_loader, desc='Testing', unit='steps')
    model.eval()
    for step, inp in enumerate(pbar):
        with torch.no_grad():

            images, cumulants, snrs, labels = inp

            images = images.to(device, non_blocking=True)
            cumulants = cumulants.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)

            logits = model(images, cumulants)

            pred = torch.argmax(logits, dim=-1)

            lab = torch.cat([lab, labels])
            scores = torch.cat([scores, pred])
            snr.extend(snrs)

    ids = {mod: [d == dataset.int_labels_map[mod] for d in lab] for mod in dataset.int_labels_map.keys()}
    for mod in ids.keys():
        scores_ = torch.tensor([s for i, s in enumerate(scores) if ids[mod][i].item() is True])
        lab_ = torch.tensor([s for i, s in enumerate(lab) if ids[mod][i].item() is True])
        snr_ = [s for i, s in enumerate(snr) if ids[mod][i].item() is True]
        print(mod)
        res = lab_ == scores_
        s = ['0_db', '5_db', '10_db', '15_db']
        idx = {i: [d == i for d in snr_] for i in s}
        acc = {i: sum(res[idx[i]]) / sum(idx[i]) for i in s}
        # accuracy = torch.sum(lab_ == scores_) / scores_.nelement()
        # print(f'Accuracy: {accuracy}')
        print(acc)
    return scores, lab, snr


if __name__ == '__main__':
    pred, lab, snr = test()