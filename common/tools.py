from datetime import datetime
from time import time
import torch
class TrainingHistory:
    def __init__(self, model_name, num_class):
        self.name = model_name
        self.train_date = datetime.now().strftime("%Y-%m-%d")
        self.start_time = time()
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.scores = torch.empty(0, dtype=torch.int)
        self.labels = torch.empty(0, dtype=torch.int)
        self.num_class = num_class

    def gather_output(self, labels, logits):
        self.labels = torch.cat((self.labels, labels))

        scores = torch.softmax(logits, dim=-1).argmax(dim=-1)
        self.scores = torch.cat((self.scores, scores))

    def end_epoch(self, loss, train=True):
        if train:
            self.losses.append(loss)
        else:
            self.val_losses.append(loss)

