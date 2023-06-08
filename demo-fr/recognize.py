from model import vargface
import torch
import os
from PIL import Image
from face_alignment import align
import numpy as np
import cv2
import pickle
from utils.data_utils import extract_deep_feature
from utils.io import FPS
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

def cosine_similarity(feats, known_feats):
    # print(torch.cat(feats).shape)
    # print(torch.cat(known_feats).T.shape)
    similarity_scores = torch.cat(feats) @ torch.cat(known_feats).T
    scores = torch.max(similarity_scores, axis=1)
    inds = torch.argsort(similarity_scores, axis=1)[:, -1]
    return similarity_scores, scores, inds

def process_image(frame):
    # Find all the faces and face encodings in the current frame of video
    aligned_rgb_img, bboxes, img = align.get_aligned_face(rgb_small_frame)
    img = np.array(img)
    
    if aligned_rgb_img is  None:
        return None, None, frame
    
    face_names = []
    for face in aligned_rgb_img:
        # aligned_rgb_img[i].show()
        # face.show()
        rgb_tensor_input = to_input(face).to(device)
        
        feature, _ = model(rgb_tensor_input)
        fliped_image = torch.flip(rgb_tensor_input, dims=[3])
        fliped_image = fliped_image.to(device)
        flipped_feature, flipped_ = model(fliped_image)
        feature = extract_deep_feature(feature, _, flipped_feature, flipped_)
        feature = [feature.detach().cpu()]
        
        # See if the face is a match for the known face(s)
        similarity_score, score, ind = cosine_similarity(feature, known_faces)
        # print(score.values.item())
        if score.values.item() < threshold: 
            name = f"Unknown: {int(((float(score.values.item())+1.0)/2.0)*100):2d}%"
        else:
            name = f"{known_names[ind]}: {int(((float(score.values.item())+1.0)/2.0)*100):2d}%"

        # print(name)
        face_names.append(name)
    return face_names, bboxes, frame

def process_result(face_names, bboxes, frame):
    if face_names is None or bboxes is None:
        return frame
    for box, name in zip(bboxes, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        # Draw a box around the face
        cv2.rectangle(frame, (int(box[0])*4, int(box[1])*4), (int(box[2])*4, int(box[3])*4), (0, 0, 255), 2)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, ((int(box[0])*4) + 6, (int(box[1])*4) - 6), font, 0.4, (255, 255, 255), lineType=cv2.LINE_AA)
    return frame

if __name__ == '__main__':
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    threshold = 0.3
    fps = FPS()
    
    # Get face gallery
    face_data = pickle.load(open('demo-fr/test_images/face_db/face_db.pkl', 'rb'))
    data = face_data['embeddings']
    known_faces = [embedding for embedding in data]
    known_names = face_data['ids']
    
    # Initialize model
    resume = "demo-fr/model/VarGFace_AdaFace.ckpt"
    model = load_pretrained_model(resume)
    feature, norm = model(torch.randn(2,3,112,112).to(device))

    # Get a reference to webcam #0 (the default one)
    process_this_frame = True
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    if not video_capture.isOpened():
        print("Failed to open webcam.")
        exit()

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Start count fps
            fps.start()
            # Preprocess frame
            face_names, bboxes, frame = process_image(frame)

        process_this_frame = not process_this_frame

        # Preprocess results
        frame = process_result(face_names, bboxes, frame)
        # Stop count fps
        fps.stop()
        # Display the resulting image
        cv2.imshow('Real-time Face Recognition', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
