import torch
import cv2
import numpy as np

def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm

def fuse_features_with_norm(stacked_embeddings, stacked_norms):
    assert stacked_embeddings.ndim == 3 # (n_features_to_fuse, batch_size, channel)
    assert stacked_norms.ndim == 3 # (n_features_to_fuse, batch_size, 1)
    
    pre_norm_embeddings = stacked_embeddings * stacked_norms
    fused = pre_norm_embeddings.sum(dim=0)
    fused, fused_norm = l2_norm(fused, axis=1)

    return fused, fused_norm 

def extract_deep_feature(embedding, _, flipped_embedding, flipped_):
    stacked_embedding = torch.stack([embedding, flipped_embedding], dim=0)
    stacked_norm = torch.stack([_, flipped_], dim=0)
    embedding, norm = fuse_features_with_norm(stacked_embedding, stacked_norm)
    embedding = embedding.data.cpu().numpy()

    return embedding

def preprocess_img(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img = np.resize(img, (112, 112))
    if len(img.shape) == 2:
        img = np.stack([img] * 3, 2)
    img = (img - 127.5) / 128.0
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)

    return img