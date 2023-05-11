import os

BASE_PATH = '../AutoModClass/AutoModClass/hybrid_dataset/training/images'
MODS = os.listdir(BASE_PATH)
SNRS = [0, 5, 10, 15]

VAL_RATIO = 0.2

BATCH_SIZE = 100
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30

MODEL_NAME = "mmnet_full"