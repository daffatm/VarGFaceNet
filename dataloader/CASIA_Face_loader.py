import numpy as np
import imageio
import os
from sklearn import preprocessing
import torch
from dataloader.augmenter import Augmenter
from torchvision.transforms import ToPILImage
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("..")

class CASIA_Face():
    def __init__(self, root, augmenter=None):
        self.image_list = []
        self.label_list = []
        self.augmenter = augmenter # Augmenter(0.2, 0.2, 0.2)

        for r, _, files in os.walk(root):
            for f in files:
                self.image_list.append(os.path.join(r, f))
                self.label_list.append(os.path.basename(r))

        le = preprocessing.LabelEncoder()
        self.label_list = le.fit_transform(self.label_list)
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = imageio.imread(img_path)
        # img = np.resize(img, (112, 112))

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)

        # Apply augmentations if augmenter is provided
        if self.augmenter != None:
            img = ToPILImage()(img)
            img = self.augmenter.augment(img)
            img = np.array(img)

        flip = np.random.choice(2)*2-1
        img = img[:, ::flip, :]
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        

        return img, target

    def __len__(self):
        return len(self.image_list)