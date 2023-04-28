import torch
import numpy as np
from core import model
from data_utils import extract_deep_feature, preprocess_img
from torch.profiler import profile, record_function, ProfilerActivity

@torch.no_grad()
def face_verification(file1, file2, resume=None, gpu=False):
    net = model.VarGFaceNet()
    if gpu:
        net = net.cuda()
    if resume:
        ckpt = torch.load(resume, map_location='cpu')
        net.load_state_dict(ckpt['net_state_dict'])
    net.eval()

    # load img1
    img1 = preprocess_img(file1)

    # load img2
    img2 = preprocess_img(file2)

    if gpu:
        img1 = img1.cuda()
        img2 = img2.cuda()
    
    # Ekstraksi embedding dari gambar pertama
    embedding, _ = net(img1)
    fliped_image = torch.flip(img1, dims=[3])
    if gpu:
        fliped_image = fliped_image.cuda()

    flipped_embedding, flipped_ = net(fliped_image)
    embedding1 = extract_deep_feature(embedding, _, flipped_embedding, flipped_)

    # Ekstraksi embedding dari gambar kedua
    embedding, _ = net(img2)
    fliped_image = torch.flip(img2, dims=[3])
    if gpu:
        fliped_image = fliped_image.cuda()
        
    flipped_embedding, flipped_ = net(fliped_image)
    embedding2 = extract_deep_feature(embedding, _, flipped_embedding, flipped_)

    # Menghitung jarak antara kedua embedding
    distance = np.linalg.norm(embedding1 - embedding2)

    # Menentukan threshold
    threshold = 0.1     #90%

    print(f"Jarak Embedding: {distance}")

    # Mengecek apakah jarak kurang dari threshold atau tidak
    if distance < threshold:
        print("Gambar tersebut adalah gambar dari orang yang sama")
    else:
        print("Gambar tersebut adalah gambar dari orang yang berbeda")

@torch.no_grad()
def inference(file, speed_mem_eval=False, resume=None, gpu=False):
    net = model.VarGFaceNet()
    if gpu:
        net = net.cuda()
    if resume:
        ckpt = torch.load(resume, map_location='cpu')
        net.load_state_dict(ckpt['net_state_dict'])
    net.eval()

    # load img
    img1 = preprocess_img(file)

    if gpu:
        img1 = img1.cuda()
        
    if speed_mem_eval:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True, profile_memory=True, with_flops=True) as prof:
            with record_function("inference"):
                # Ekstraksi embedding
                embedding, _ = net(img1)
                fliped_image = torch.flip(img1, dims=[3])
                if gpu:
                    fliped_image = fliped_image.cuda()

                flipped_embedding, flipped_ = net(fliped_image)
                embedding = extract_deep_feature(embedding, _, flipped_embedding, flipped_)
                
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_memory_usage", row_limit=10))
    else:
        # Ekstraksi embedding
        embedding, _ = net(img1)
        fliped_image = torch.flip(img1, dims=[3])
        if gpu:
            fliped_image = fliped_image.cuda()

        flipped_embedding, flipped_ = net(fliped_image)
        embedding = extract_deep_feature(embedding, _, flipped_embedding, flipped_)
        
        return embedding