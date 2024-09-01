from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
import dlib
from imutils import face_utils


def generate_landmark(image_list, save_path, face_detector, face_predictor, period=1, num_frames=10):

    for img_path in tqdm(image_list):
        frame_org = cv2.imread(img_path)
        frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)

        faces = face_detector(frame, 1)
        if len(faces) == 0:
            tqdm.write('No faces in {}'.format(img_path))
            continue
        face_s_max = -1
        landmarks = []
        size_list = []
        for face_idx in range(len(faces)):
            landmark = face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0, y0 = landmark[:, 0].min(), landmark[:, 1].min()
            x1, y1 = landmark[:, 0].max(), landmark[:, 1].max()
            face_s = (x1 - x0) * (y1 - y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        landmarks = np.concatenate(landmarks).reshape((len(size_list),) + landmark.shape)
        landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]

        save_path_ = save_path + img_path.split('/')[-2] + '/'
        os.makedirs(save_path_, exist_ok=True)

        land_path = os.path.join(save_path_, os.path.basename(img_path).replace('jpg', 'npy'))
        np.save(land_path, landmarks)

    return


if __name__ == '__main__':
    root = '/home/og/home/lqx/datasets/FF++/original_sequences/youtube/c23/frames/'

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = '/home/og/home/lqx/code/FaceAdaptation/utils/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

    save_path = root.replace('frames', 'landmarks')
    vids = os.listdir(root)
    images = glob(os.path.join(root, '*/*.jpg'))

    generate_landmark(images, save_path, face_detector, face_predictor)