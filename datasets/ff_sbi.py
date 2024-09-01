import os
import json
import numpy as np
import glob
import PIL.Image as I
import torch
import torchvision.transforms
import torchvision.transforms as tfm
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from datasets.sbi import SBITransform
from datasets.dct_transforms import get_dct_transform

import matplotlib.pyplot as plt


class FFSBIDatasets(Dataset):
    def __init__(self, cfg, mode='train', quality='c40'):
        super(FFSBIDatasets, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.test_imagenum_per_vid = self.cfg.TESTING.IMAGENUM_PER_VID

        # get video list
        mode_list = get_model_list(self.cfg.DATAS.ROOT, mode)
        mode_list.sort()
        mode_list = choice_vids(mode_list, split_ratio=1.0)
        self.mode_list = mode_list

        self.length_ratio = 1
        crop_face_images = get_face_images(self.cfg.DATAS.ROOT, mode_list=mode_list, quality=quality)
        self.face_data = crop_face_images
        # transforms
        self.sbi_trans = SBITransform(quality=quality)

        self.mean = cfg.AUGS.MEAN
        self.std = cfg.AUGS.STD

        self.to_tensor = tfm.Compose([
            tfm.RandomHorizontalFlip(cfg.AUGS.FLIP_PROB),
            tfm.ToTensor(),
            tfm.Normalize(mean=self.mean, std=self.std)
        ])
        if self.cfg.DATAS.WITH_FREQUENCY:
            self.freq_dct = get_dct_transform(cfg)
        else:
            self.freq_dct = None

    def __len__(self):
        if self.mode == 'test':
            return len(self.mode_list)
        else:
            return len(self.face_data)

    def __getitem__(self, idx):
        # Frame level
        fname, label = self.face_data[idx]
        img_r, img_f = self.sbi_trans(fname)
        img_f = I.fromarray(img_f)
        img_r = I.fromarray(img_r)
        if self.freq_dct is not None:
            img_r_freq = self.freq_dct(img_r).float()
            img_f_freq = self.freq_dct(img_f).float()
        img_f = self.to_tensor(img_f)
        img_r = self.to_tensor(img_r)

        if self.freq_dct is not None:
            return img_r, img_r_freq, img_f, img_f_freq
        else:
            return img_r, img_f


def choice_vids(mode_list, split_ratio):
    if split_ratio == 1.0:
        choice_list = mode_list
    else:
        vid_num = round(len(mode_list) * split_ratio)
        choice_list = np.random.choice(mode_list, vid_num, replace=False)
    print("Choice item : {}/{}".format(len(choice_list), len(mode_list)))
    return choice_list


# crop faces according to the face bounding box [x,y,w,h]
def crop_retina_faces(img, box, expand_ratio=1.3):
    x,y,w,h = box
    c_x, c_y = x + w/2, y+h/2
    new_w, new_h = w*expand_ratio, h*expand_ratio
    left = max(0, c_x - new_w/2)
    top = max(0, c_y - new_h/2)
    # torchvision transforms using top(y) left(x) height(h) weight(w) to crop image
    img = F.crop(img, round(top), round(left), round(new_h), round(new_w))
    return img


def choice_frames(fname_list, split_ratio):
    fname_list = np.array(fname_list)
    if split_ratio == 1.0:
        choice_list = fname_list
    else:
        frame_num = round(len(fname_list) * split_ratio)
        choice_list = np.random.choice(fname_list, frame_num, replace=False)
    print("Choice item : {}/{}".format(len(choice_list), len(fname_list)))
    return choice_list


def face_filter(fname):

    face_file = fname.replace(fname[-4:], '.npy').replace('/frames/', '/retinaface/')
    if os.path.exists(face_file):
        return True
    else:
        return False


def get_face_images(root, mode_list, quality='c40'):
    if quality == 'c40':
        suffix = 'png'
    elif quality == 'c23':
        suffix = 'jpg'
    else:
        suffix = 'png'
    all_real_images = glob.glob(os.path.join(root, 'original_sequences', 'youtube', quality, 'frames/*/*.' + suffix))
    all_real_images = [fname for fname in all_real_images if mode_filter(fname, mode_list)]
    all_real_images = [fname for fname in all_real_images if face_filter(fname)]
    all_real_images = [(fname, 0) for fname in all_real_images if landmark_filter(fname, suffix)]
    return all_real_images


def get_model_list(root, mode):
    json_file = os.path.join(root, 'splits', mode + '.json')
    mode_list = []
    json_data = json.load(open(json_file, 'r'))

    for d in json_data:
        mode_list += d

    if mode == 'test':
        for d in json_data:
            fake_vid = [d[0]+'_'+d[1], d[1]+'_'+d[0]]
            mode_list += fake_vid
    return mode_list


def mode_filter(fname, mode_list):
    vid_num = fname.split('/')[-2]
    vid_num = vid_num[:3]
    return vid_num in mode_list


def method_filter(fname, method='Deepfakes'):
    return method in fname or 'original' in fname


def quality_filter(fname, quality='c40'):
    return quality in fname


def label_mapper(fname):
    label = 0 if 'original' in fname else 1
    return fname, label


def landmark_filter(fname, suffix):
    landmark_file = fname.replace(suffix, 'npy').replace('/frames/', '/landmarks/')
    return os.path.exists(landmark_file)


if __name__ == '__main__':
    from configs.defaults import get_config
    from datasets.trans import get_transfroms
    cfg = get_config()
    print(cfg.DATAS.WITH_FREQUENCY)
    trans = get_transfroms(cfg, mode='train')
    file = '/home/og/home/lqx/code/FaceAdaptation/configs/yamls/final.yaml'
    cfg.merge_from_file(file)
    cfg.DATAS.WITH_FREQUENCY = True
    cfg.DATAS.SOURCE = 'NeuralTextures'
    # cfg.DATAS.TARGET = ['Deepfakes', 'Face2Face', 'FaceSwap']
    cfg.DATAS.TARGET = ['FaceSwap']

    source_dp = FFSBIDatasets(cfg, mode='train', quality='c23')
    data_loader = torch.utils.data.DataLoader(source_dp, batch_size=2, num_workers=8, shuffle=True)
    all_freq = []
    for data in data_loader:
        a,b,c,d = data
        print(torch.max(b))
        print('done')
        # all_freq.append(freq)


