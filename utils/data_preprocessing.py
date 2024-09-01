import cv2
import os
import numpy as np
import dlib
from glob import glob
from tqdm import tqdm

import torch.nn as nn
nn.MaxPool2d(kernel_size=2)
import torch.nn.functional as F


def set_face_detector(mode='front'):
    if mode == 'front':
        face_detector = dlib.get_frontal_face_detector()
    if mode == 'cnn':
        if not dlib.DLIB_USE_CUDA:
            print('@ dlib is not using CUDA.')
        face_detector = dlib.cnn_face_detection_model_v1('/dlib_model/mmod_human_face_detector.dat')
    return face_detector


def parse_vid(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frame_num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = np.int32(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    imgs = []
    while True:
        success, image = vidcap.read()
        if success:
            imgs.append(image)
        else:
            break

    vidcap.release()
    if len(imgs) != frame_num:
        frame_num = len(imgs)
    return imgs


def crop(im, left, top, right, bottom):
    pad = 0.4
    pad_t = 0.5
    left_ = np.maximum(int(left - pad * (right - left)), 0)
    top_ = np.maximum(int(top - pad_t * (bottom - top)), 0)
    right_ = int(right + pad * (right - left))
    bottom_ = int(bottom + pad * (bottom - top))
    roi = im[top_:bottom_, left_:right_]
    return roi


def process_celebDF(data_root):
    face_detector = set_face_detector('front')
    version = ['Celeb-DF-v1', 'Celeb-DF-v2']

    for ver in version:
        celeb_data_root = data_root + ver
        test_list = []
        anno_path = os.path.join(celeb_data_root, 'List_of_testing_videos.txt')
        with open(anno_path, 'r') as f:
            for line in f:
                lb = line.split(' ')[0]
                name = line.replace('\n', '').split(' ')[1]
                name = name.replace('.mp4', '')
                test_list.append(name)

        for fname in test_list:
            vidpath = os.path.join(celeb_data_root, fname + '.mp4')
            foldername = os.path.basename(fname)
            folderpath = os.path.join(celeb_data_root, fname)
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
            imgs = parse_vid(vidpath)
            print(folderpath)

            for frame_id in tqdm(range(len(imgs))):
                im = imgs[frame_id]
                faces = face_detector(np.uint8(im))
                if faces is not None or len(faces) > 0:
                    for i, d in enumerate(faces):
                        try:
                            left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
                        except:
                            left, top, right, bottom = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()

                roi = crop(im, left, top, right, bottom)
                cv2.imwrite(folderpath + '/{}_{:06d}.jpg'.format(foldername, frame_id), roi)
                cv2.imwrite('tmp_celeb.jpg'.format(foldername, frame_id), roi)


if __name__ == '__main__':
    celeb_df_root = '/root/workdir/datasets/Celeb-DF/'
    process_celebDF(celeb_df_root)
