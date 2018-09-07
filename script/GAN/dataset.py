import os, random
from PIL import Image
import cv2, math
from torchvision import transforms
import numpy as np
import torch
from config import cfg
import matplotlib.pyplot as plt
from itertools import permutations as P


def train_transform():
    img = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return img


def val_transform():
    img = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5, 0.5, 0.5))
    ])
    return img

# add random erasing
def train_loader(path):

    img = Image.open(path).convert('RGB')

    # add random noise
    if random.uniform(0,1) > 0.5:
        return img

    sl = 0.02
    sh = 0.4
    r1 = 0.3
    # mean=[0.4914, 0.4822, 0.4465]
    mean = [0.0, 0.0, 0.0]
    img = np.array(img)
    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1/r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            if img.shape[2] == 3:
                img[0, x1:x1+h, y1:y1+w] = mean[0]
                img[1, x1:x1+h, y1:y1+w] = mean[1]
                img[2, x1:x1+h, y1:y1+w] = mean[2]
            else:
                img[0, x1:x1+h, y1:y1+w] = mean[0]
            return Image.fromarray(np.uint8(img))

    return Image.fromarray(np.uint8(img))


def val_loader(path):
    img = Image.open(path).convert('RGB')

    return img


class Market_DataLoader():
    def __init__(self, imgs_path, pose_path, idx_path, transform, loader):
        # train/test index
        lines = open(idx_path, 'r').readlines()
        idx = [int(line.split()[0]) for line in lines]

        # images
        images = os.listdir(imgs_path)
        poses = os.listdir(pose_path)
        images = [im for im in images if int(im.split('_')[0]) in idx and im in poses]

        # pairs
        data = []
        for i in idx:
            tmp = [j for j in images if int(j.split('_')[0]) == i]
            data += list(P(tmp, 2))

        random.shuffle(data)
        # self.data = data[0:5000*cfg.TRAIN.BATCH_SIZE]
        self.data = data
        self.imgs_path = imgs_path
        self.pose_path = pose_path
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        src_path, tgt_path = self.data[index]

        src_img = self.loader(os.path.join(self.imgs_path, src_path))
        tgt_img = self.loader(os.path.join(self.imgs_path, tgt_path))
        pose    = self.loader(os.path.join(self.pose_path, tgt_path))

        src_img, tgt_img, pose = self.transform(src_img), self.transform(tgt_img), self.transform(pose)

        return src_img, tgt_img, pose

    def __len__(self):
        return len(self.data)


class Market_test():
    def __init__(self, imgs_name, imgs_path, pose_path, transform, loader):
        # images
        images = imgs_name
        poses = os.listdir(pose_path)

        # pairs
        data = []
        for i in images:
            for j in poses:
                name = i.split('.')[0] + '_to_' + j.split('.')[0]
                data.append([i, j, name])

        self.data = data
        self.imgs_path = imgs_path
        self.pose_path = pose_path
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        src_path, pose, name = self.data[index]

        src_img = self.loader(os.path.join(self.imgs_path, src_path))
        pose    = self.loader(os.path.join(self.pose_path, pose))

        src_img, pose = self.transform(src_img), self.transform(pose)

        return src_img, pose, name

    def __len__(self):
        return len(self.data)