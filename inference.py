import torch
import numpy as np
from core import model
from data_utils import extract_deep_feature, preprocess_img
from torch.profiler import profile, record_function, ProfilerActivity

def load_model(resume=None, gpu=False):
    net = model.VarGFaceNet()
    if gpu:
        net = net.cuda()
    if resume:
        ckpt = torch.load(resume, map_location='cpu')
        net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    
    return net

@torch.no_grad()
def extract_emb(net, img, gpu, flip=True):
    embedding, _ = net(img)
    
    if flip:
        fliped_image = torch.flip(img, dims=[3])
        if gpu:
            fliped_image = fliped_image.cuda()

        flipped_embedding, flipped_ = net(fliped_image)
        embedding = extract_deep_feature(embedding, _, flipped_embedding, flipped_)
        embedding = torch.from_numpy(embedding).float()
    
    return embedding

@torch.no_grad()
def face_verification(file1, file2, resume=None, gpu=False):
    net = load_model(resume, gpu)
    embedding, _ = net(torch.randn(2,3,112,112))
        
    features = []
    imgs = [file1, file2]
    # Ekstrak embedding
    for img in imgs:
        # load 
        img = preprocess_img(img)
        if gpu:
            img = img.cuda()

        embedding = extract_emb(net, img, gpu) 
        features.append(embedding)

    # Mengitung score kemiripan
    similarity = torch.cat(features) @ torch.cat(features).T
    # Menentukan threshold
    threshold = 0.6     #60% kemiripan
    print(f"Jarak Embedding: {similarity[0][1]}")

    # Mengecek apakah jarak kurang dari threshold atau tidak
    if similarity[0][1] > threshold:
        print("Gambar tersebut adalah gambar dari orang yang sama")
    else:
        print("Gambar tersebut adalah gambar dari orang yang berbeda")

@torch.no_grad()
def inference(file, speed_mem_eval=False, resume=None, gpu=False):
    net = load_model(resume, gpu)
    embedding, _ = net(torch.randn(2,3,112,112))

    # load img
    img = preprocess_img(file)
    if gpu:
        img = img.cuda()
        
    if speed_mem_eval:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True, profile_memory=True, with_flops=True) as prof:
            with record_function("extract_emb"):
                # Ekstraksi embedding
                embedding = extract_emb(net, img, gpu, flip=False) 
                
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_memory_usage", row_limit=10))
    else:
        # Ekstraksi embedding
        embedding = extract_emb(net, img, gpu) 
        return embedding