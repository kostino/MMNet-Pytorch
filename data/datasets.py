import os
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np


class ModulationsDataset(Dataset):
    def __init__(
            self,
            path: str,
            modulations: List[str],
            snrs: List[int],
            from_directory: bool = True
    ):
        self.path = path
        self.mods = modulations
        self.snr_fs = [str(snr) + "_db" for snr in snrs]
        self.images = []
        self.labels = []
        self.cumulants = []
        self.snrs = []
        self.int_labels_map = {mod: self.mods.index(mod) for mod in self.mods}

        if from_directory:
            self._generate_from_directory()
        else:
            self._generate_from_file()

    def _generate_from_directory(self):
        for mod in self.mods:
            for snr in self.snr_fs:
                path = os.path.join(self.path, mod, snr)
                images = [os.path.join(path,img) for img in os.listdir(path)]
                cumulants = [p.replace('images', 'cumulants').replace('.png', '.cum') for p in images]
                labels = [mod for _ in images]
                snrs = [snr for _ in images]

                self.images.extend(images)
                self.cumulants.extend(cumulants)
                self.snrs.extend(snrs)
                self.labels.extend(labels)

    def _generate_from_file(self):
        with open(self.path, "r") as fp:
            for line in fp.readlines():
                ll = line.rstrip()
                image, cumulants, snr, label = ll.split(' ')
                self.images.append(image)
                self.cumulants.append(cumulants)
                self.snrs.append(snr)
                self.labels.append(label)

    @staticmethod
    def _load_cumulants(path):
        cumulants = np.fromfile(path, np.complex128)
        real = cumulants.real
        imaginary = cumulants.imag
        cum = torch.tensor(np.concatenate((real, imaginary)))
        return cum

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image_path = self.images[idx]
        cum_path = self.cumulants[idx]

        cumulants = self._load_cumulants(cum_path)
        image = to_tensor(Image.open(image_path))
        label = self.int_labels_map[self.labels[idx]]
        snr = self.snrs[idx]

        return image, cumulants, snr, label


if __name__ == "__main__":
    path = '../../AutoModClass/AutoModClass/hybrid_dataset/training/images'
    mods = os.listdir(path)
    snrs = [0, 5, 10, 15]
    dataset = ModulationsDataset(path, mods, snrs)
    a = dataset[0]