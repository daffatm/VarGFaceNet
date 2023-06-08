import sys
import os
import time
import numpy as np
import scipy.io
import os
import torch.utils.data
from core import model
from dataloader.LFW_loader import LFW
from config import LFW_DATA_DIR
from data_utils import extract_deep_feature, preprocess_img
from torch.profiler import profile, record_function, ProfilerActivity

def parseList(root, name='lfw-112x112'):
    with open(os.path.join(root, 'pairs.txt')) as f:
        pairs = f.read().splitlines()[1:]
    
    # ORG
    folder_name = name
    nameLs = []
    nameRs = []
    folds = []
    flags = []
    for i, p in enumerate(pairs):
        p = p.split('\t')
        if len(p) == 3:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[2])))
            fold = i // 600
            flag = 1
        elif len(p) == 4:
            nameL = os.path.join(root, folder_name, p[0], p[0] + '_' + '{:04}.jpg'.format(int(p[1])))
            nameR = os.path.join(root, folder_name, p[2], p[2] + '_' + '{:04}.jpg'.format(int(p[3])))
            fold = i // 600
            flag = -1
        nameLs.append(nameL)
        nameRs.append(nameR)
        folds.append(fold)
        flags.append(flag)

    return [nameLs, nameRs, folds, flags]

def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    accuracy = (p + n) * 1.0 / len(scores)
    return accuracy

def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def evaluation_10_fold(root='./result/pytorch_result.mat'):
    ACCs = np.zeros(10)
    thresholds = np.zeros(10)
    result = scipy.io.loadmat(root)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        if featureLs[valFold[0], :].shape[0] == 0:
            continue
        
        if featureRs[valFold[0], :].shape[0] == 0:
            continue
        
        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        # if len(scores)==0 : 
        #     continue;

        thresholds[i] = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], thresholds[i])
        
    return ACCs, thresholds

@torch.no_grad()
def getFeatureFromTorch(lfw_dir, name, feature_save_dir, resume=None, gpu=True):
    net = model.VarGFaceNet()
    if gpu:
        net = net.cuda()
    if resume:
        ckpt = torch.load(resume, map_location='cpu')
        net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    nl, nr, flods, flags = parseList(lfw_dir, name)
    lfw_dataset = LFW(nl, nr)

    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=32,
                                              shuffle=False, num_workers=4, drop_last=False)

    featureLs = None
    featureRs = None
    count = 0

    # print(lfw_loader.dataset.__len__())
    for data in lfw_loader:
        if gpu:
            for i in range(len(data)):
                data[i] = data[i].cuda()
        
        count += data[0].size(0)
        sys.stdout.write("\rextracing deep features from the face pair {}/{}".format(count, lfw_loader.dataset.__len__()))
        sys.stdout.flush()
        res = []
        for d in data:
            out, norms = net(d)
            fliped_image = torch.flip(d, dims=[3])
            flipped_embedding, flipped_ = net(fliped_image)
            embedding = extract_deep_feature(out, norms, flipped_embedding, flipped_)
            res.append(embedding)
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        # featureLs.append(featureL)
        # featureRs.append(featureR)
        
    if not os.path.exists('./result'):
        os.makedirs('./result')
                
    result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
    scipy.io.savemat(feature_save_dir, result)