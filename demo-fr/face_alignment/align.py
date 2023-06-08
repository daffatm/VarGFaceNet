import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
import torch
import numpy as np
from typing import Union

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn_model = mtcnn.MTCNN(device=device, crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def read_image(image: Union[np.ndarray, str, list], rgb_pil_image=None):
    if rgb_pil_image is None:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            # image_array = np.array(image)
            # shape = image_array.shape
            # print(shape)
        elif isinstance(image, list):
            image = [Image.open(str(img)).convert('RGB') for img in image]
            # image = np.stack(image)
        else:
            image = Image.fromarray(image)
            # image_array = np.array(image)
            # shape = image_array.shape
            # print(shape)
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        image = rgb_pil_image
    return image

def get_aligned_face(image_path: Union[str, np.ndarray, list], rgb_pil_image=None):
    img = read_image(image_path)
    # find face
    bboxes, faces = mtcnn_model.align_multi(img) #limit=None bisa di set ke 1
    if len(faces)>0:
        return faces, bboxes, img
    return None, None, img


