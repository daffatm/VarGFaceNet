import numpy as np
import imageio
import torch
import torch
import sys
sys.path.append("..")
# from retrieval.dataloaders.preprocessing import preprocess

class LFW():
    def __init__(self, imgl, imgr):

        self.imgl_list = imgl
        self.imgr_list = imgr

    def __getitem__(self, index):
        imgl = imageio.imread(self.imgl_list[index])
        imgl = np.resize(imgl, (112, 112))
        if len(imgl.shape) == 2:
            imgl = np.stack([imgl] * 3, 2)
        imgr = imageio.imread(self.imgr_list[index])
        imgr = np.resize(imgr, (112, 112))
        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, 2)

        # imgl = imgl[:, :, ::-1]
        # imgr = imgr[:, :, ::-1]
        imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        imgs = [torch.from_numpy(i).float() for i in imglist]
        return imgs

    def __len__(self):
        return len(self.imgl_list)