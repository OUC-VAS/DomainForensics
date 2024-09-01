import sys
sys.path.append('/home/og/home/lqx/code/FaceAdaptation')
import os
import glob
import torch
import numpy as np
import PIL.Image as I

from datasets.ff import get_model_list, get_face_images, crop_retina_faces, choice_vids, face_filter
import torchvision.transforms as tfm
from torch.utils.data import Dataset
from datasets.dct_transforms import get_dct_transform
import json


def get_dfdcp_mode_list(root, mode='train'):
    vid_list = []
    all_vid_data = json.load(open(os.path.join(root, 'dataset.json')))

    for vid_name, vid_info in all_vid_data.items():
        if vid_info['set'] == mode:
            vid_list.append((vid_name, 0 if vid_info['label'] == 'real' else 1))
    return vid_list



def face_celeb_filter(fname):
    face_file = fname.replace(fname[-4:], '.npy').replace('frames', 'retinaface')
    if os.path.exists(face_file):
        return True
    else:
        return False


def get_dfdcp_face_images(root, mode_list, mode='train'):
    all_images = []

    for (vid_name, label) in mode_list:
        frames = glob.glob(os.path.join(root, vid_name.replace('.mp4', '_frames/*.jpg')))
        if 'original' in vid_name:
            all_images += [(f, label) for f in frames] * 4
        else:
            all_images += [(f, label) for f in frames]
    return all_images


class DFDCPDatasets(Dataset):
    def __init__(self, cfg, mode='train', trans=None, split_ratio=1.0):
        super(DFDCPDatasets, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.trans = trans
        self.expand_ratio = self.cfg.DATAS.EXPAND_RATIO
        self.root = '/home/og/home/lqx/mnt_point/datasets/DFDCP_password_123456/DFDCP/'

        self.mode_list = get_dfdcp_mode_list(self.root, mode)
        self.mode_list = choice_vids(self.mode_list, split_ratio=split_ratio)
        self.face_data = get_dfdcp_face_images(self.root, self.mode_list, mode=mode)
        self.expand_ratio = self.cfg.DATAS.EXPAND_RATIO
        self.mean = cfg.AUGS.MEAN
        self.std = cfg.AUGS.STD

        self.to_tensor = tfm.Compose([
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

    def stat_images(self):
        # print(self.face_data)
        real_num = len([_ for _ in self.face_data if 0 == _[1]])
        fake_num = len([_ for _ in self.face_data if 1 == _[1]])
        print("Total real num is : ", real_num, ", Total fake num is : ", fake_num)

    def __getitem__(self, idx):
        if self.mode == 'test':
            # Video level
            vid_num = self.mode_list[idx]
            # fnames = np.array([fn for fn in self.face_data if vid_num == fn[0].split('/')[-2]])
            fnames = glob.glob(os.path.join(self.root, vid_num[0].replace('.mp4', '_frames/*.jpg')))
            images = []
            imgs_freq = []

            label = int(vid_num[1])
            for _ in fnames:
                box = np.load(_.replace(_[-4:], '.npy').replace('frames', 'retinaface'))
                img = I.open(_)
                img = crop_retina_faces(img, box, expand_ratio=self.expand_ratio)
                if self.trans is not None:
                    img_trans = self.trans(img)
                    if self.freq_dct is not None:
                        img_freq = self.freq_dct(img_trans).float()
                        imgs_freq.append(img_freq.unsqueeze(0))
                    img_trans = self.to_tensor(img_trans)
                    images.append(img_trans.unsqueeze(0))
            images = torch.cat(images, dim=0)
            if self.freq_dct is not None:
                imgs_freq = torch.cat(imgs_freq, dim=0)
                return images, imgs_freq, label
            else:
                return images, label
        else:
            # Frame level
            fname, label = self.face_data[idx]
            box = np.load(fname.replace(fname[-4:], '.npy').replace('frames', 'retinaface'))
            # open iamge
            img = I.open(fname)
            img = crop_retina_faces(img, box, expand_ratio=self.expand_ratio)
            if self.trans is not None:
                if self.freq_dct is None:
                    img_trans = self.trans(img)
                    img_trans = self.to_tensor(img_trans)
                    return img_trans, label
                else:
                    img_trans = self.trans(img)
                    img_freq = self.freq_dct(img_trans).float()
                    img_trans = self.to_tensor(img_trans)
                    return img_trans, img_freq, label
            else:
                img = self.temp_trans(img)
                img = np.asarray(img)
                return img, label


if __name__ == '__main__':
    root = '/home/og/home/lqx/datasets/Celeb-DF/Celeb-DF-v2'
    from configs.defaults import get_config
    from datasets.trans import get_transfroms
    cfg = get_config()
    trans = get_transfroms(cfg, mode='train')
    file = '/home/og/home/lqx/code/FaceAdaptation/configs/yamls/final.yaml'
    cfg.merge_from_file(file)
    cfg.DATAS.WITH_FREQUENCY = True
    cfg.DATAS.ROOT = root

    source_dp = DFDCPDatasets(cfg, mode='train', trans=trans)
    source_dp.stat_images()

    print(source_dp[0])
    # data_loader = torch.utils.data.DataLoader(source_dp, batch_size=32, num_workers=8)
    # all_freq = []
    # for data in data_loader:
    #     _, freq, _ = data
    #     print(torch.max(freq))
    #     print('done')
        # all_freq.append(freq)

