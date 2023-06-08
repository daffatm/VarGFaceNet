from model import vargface
import torch
import os
import pickle
from face_alignment import align
import numpy as np
from utils.data_utils import extract_deep_feature
import warnings

warnings.filterwarnings("ignore", category=Warning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_pretrained_model(resume=None):
    # load model and pretrained statedict
    model = vargface.VarGFaceNet().to(device)
    ckpt = torch.load(resume, map_location='cpu')
    model.load_state_dict(ckpt['net_state_dict'])
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    rgb_img = ((np_img / 255.) - 0.5) / 0.5
    tensor = torch.tensor([rgb_img.transpose(2,0,1)]).float()
    return tensor

if __name__ == '__main__':
    resume = "demo-fr/model/VarGFace_AdaFace.ckpt"
    model = load_pretrained_model(resume)
    feature, norm = model(torch.randn(2,3,112,112).to(device))

    test_image_path = 'demo-fr/test_images/face_db'
    face_data = {"feat": [], "id": []}
    # print(len(sorted(os.listdir(test_image_path))))
    for fname in sorted(os.listdir(test_image_path)):
        file = os.path.splitext(fname)[1]
        if file == ".pkl":
            continue
        identities = os.path.splitext(fname)[0]
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img, bboxes, img = align.get_aligned_face(path)
        rgb_tensor_input = to_input(aligned_rgb_img[0]).to(device)
        
        feature, _ = model(rgb_tensor_input)
        fliped_image = torch.flip(rgb_tensor_input, dims=[3])
        fliped_image = fliped_image.to(device)
        flipped_feature, flipped_ = model(fliped_image)
        feature = extract_deep_feature(feature, _, flipped_feature, flipped_)
        
        face_data['feat'].append(feature.detach().cpu())
        face_data['id'].append(identities)
        
    output = "demo-fr/test_images/face_db/face_db.pkl"
    pickle.dump({'embeddings': face_data['feat'], 'ids': face_data['id']}, open(output, 'wb'))
    print(f"Resgistrasi {len(face_data['id'])} wajah berhasil..")
    
