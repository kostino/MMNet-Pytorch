if __name__ == "__main__":
    from tqdm import tqdm

    from torch.optim import SGD, Adam
    from torch.optim.lr_scheduler import StepLR
    from torch.nn import CrossEntropyLoss
    from torch.profiler import profile, record_function, ProfilerActivity
    from models.fifnet import FiFNet
    from models.ccnn import CCNN
    from models.drcnn import DrCNN
    from common.utils import update_config, AverageMeter
    from common.config import _C as cfg
    from common.train_utils import *
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-cfg", "--cfg", type=str, default='./experiments/DrCNN.yaml')

    args = parser.parse_args()

    cfg = update_config(cfg, args.cfg)

    device = handle_device(cfg.CUDNN)

    train_loader, val_loader = get_dataloaders(cfg)

    model = CCNN(hocs=True)
    model.to(device)

    optimizer = Adam(model.parameters(), cfg.TRAIN.LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=2, gamma=.7)
    criterion = CrossEntropyLoss()

    epoch_losses = []
    epoch_val_losses = []

    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        epoch_loss = train_one_epoch(model, optimizer, scheduler, criterion, train_loader, device, cfg, epoch)
        epoch_losses.append(epoch_loss.average())

        pbar = tqdm(val_loader, desc='Validating Epoch {}/{}'.format(epoch, cfg.TRAIN.NUM_EPOCHS), unit='steps')
        model.eval()
        epoch_val_loss = AverageMeter()
        for step, inp in enumerate(pbar):
            with torch.no_grad():
                images, cumulants, _, labels = inp

                images = images.to(device, non_blocking=True)
                cumulants = cumulants.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True)

                logits = model(images, cumulants)

            batch_loss = criterion(logits, labels)

            epoch_val_loss.update(batch_loss.item())
            pbar.set_postfix({"last_loss": batch_loss.item(), "epoch_loss": epoch_val_loss.average()})

        epoch_val_losses.append(epoch_val_loss.average())
        if epoch_val_loss.average() <= min(epoch_val_losses):
            torch.save(model.state_dict(), "./ckpt/" + cfg.MODEL.NAME)
