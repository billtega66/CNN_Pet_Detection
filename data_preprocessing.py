import kagglehub
path = kagglehub.dataset_download("tanlikesmath/the-oxfordiiit-pet-dataset")

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter, defaultdict
import shutil
import math
import random

from pathlib import Path

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

assert math.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0)

IMAGE_DIR = '/root/.cache/kagglehub/datasets/tanlikesmath/the-oxfordiiit-pet-dataset/versions/1/images'

image_paths = list(Path(IMAGE_DIR).rglob('*'))

image_paths = [
    p for p in image_paths
    if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
]


BASE_DIR = '/root/.cache/kagglehub/datasets/tanlikesmath/the-oxfordiiit-pet-dataset/versions/1/'
IMAGE_DIR = os.path.join(BASE_DIR, 'images')

OUTPUT_DIR = os.path.join(BASE_DIR, 'splits')
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR   = os.path.join(OUTPUT_DIR, 'val')
TEST_DIR  = os.path.join(OUTPUT_DIR, 'test')

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

class_to_images = defaultdict(list)

for fname in os.listdir(IMAGE_DIR):
    if fname.lower().endswith('.jpg'):
        class_name = '_'.join(fname.split('_')[:-1])
        class_to_images[class_name].append(fname)

random.seed(42)  # reproducibility

for class_name, images in class_to_images.items():
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    train_imgs = images[:n_train]
    val_imgs   = images[n_train:n_train + n_val]
    test_imgs  = images[n_train + n_val:]

    for split, split_imgs in zip(
        [TRAIN_DIR, VAL_DIR, TEST_DIR],
        [train_imgs, val_imgs, test_imgs]
    ):
        class_dir = os.path.join(split, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for img in split_imgs:
            src = os.path.join(IMAGE_DIR, img)
            dst = os.path.join(class_dir, img)
            shutil.copy(src, dst)