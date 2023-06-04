import os
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch.utils.data
from torch.nn import DataParallel
from core import model
import torchvision.transforms as transforms
import subprocess
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import tinyface_helper
# DataLoader
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from data_utils import extract_deep_feature

class ListDataset(Dataset):
    def __init__(self, img_list):
        super(ListDataset, self).__init__()
        self.img_list = img_list
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # Load Image
        image_path = self.img_list[idx]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:, :, :3]

        # To Tensor
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, idx

def prepare_dataloader(img_list, batch_size, num_workers=0):
    image_dataset = ListDataset(img_list)
    dataloader = DataLoader(image_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    
    return dataloader

# Evaluation
def infer(model, dataloader, use_flip_test=True, gpu=True):
    features = []
    with torch.no_grad():
        for images, idx in tqdm(dataloader):
            if gpu:
                images = images.cuda()

            embedding, _ = model(images)
            
            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                if gpu:
                    fliped_images = fliped_images.cuda()

                flipped_embedding, flipped_ = model(fliped_images)

                fused_feature = extract_deep_feature(embedding, _, flipped_embedding, flipped_)
                fused_feature = torch.from_numpy(fused_feature).float()
                features.append(fused_feature)
            else:
                features.append(embedding)

    features = np.concatenate(features, axis=0)
    return features

def load_model(resume, gpu=True):
    # define backbone and margin layer
    net = model.VarGFaceNet()
    ckpt = torch.load(resume, map_location='cpu')
    net.load_state_dict(ckpt['net_state_dict'])

    # gpu init
    if gpu:
        multi_gpus = False
        if torch.cuda.device_count() > 1:
            multi_gpus = True
        device = torch.device('cuda')

        if multi_gpus:
            net = DataParallel(net).to(device)
        else:
            net = net.to(device)

    net.eval()
    return net

def calc_accuracy(tinyface_test, probe, gallery, save_dir, do_norm=True):
    if do_norm: 
        probe = probe / np.linalg.norm(probe, ord=2, axis=1).reshape(-1,1)
        gallery = gallery / np.linalg.norm(gallery, ord=2, axis=1).reshape(-1,1)
        
    # Similarity
    result = (probe @ gallery.T)
    
    index = np.argsort(-result, axis=1)
    
    p_l = np.array(tinyface_test.probe_labels)
    g_l = np.array(tinyface_test.gallery_labels)
    
    acc_list = []
    for rank in [1, 5, 10, 20]:
        correct = 0
        for ix, probe_label in enumerate(p_l):
            pred_label = g_l[index[ix][:rank]]
            
            if probe_label in pred_label:
                correct += 1
                
        acc = correct / len(p_l)
        acc_list += [acc * 100]
    
    print(acc_list)
    pd.DataFrame({'rank':[1, 5, 10, 20], 'values':acc_list}).to_csv(os.path.join(save_dir, 'tinyface_result.csv'), index=False)