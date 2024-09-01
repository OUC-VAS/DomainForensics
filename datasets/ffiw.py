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



def get_ffiw_mode_list(root, mode='train'):
    vid_list = []

    if mode == 'train':
        all_real_vid = glob.glob(os.path.join(root, 'source/train/*.mp4'))
        all_fake_vid = glob.glob(os.path.join(root, 'target/train/*.mp4'))
        vid_list += [(vid_name, 0) for vid_name in all_real_vid]
        vid_list += [(vid_name, 1) for vid_name in all_fake_vid]
    else:
        all_real_vid = glob.glob(os.path.join(root, 'source/val/*.mp4'))
        all_fake_vid = glob.glob(os.path.join(root, 'target/val/*.mp4'))
        vid_list += [(vid_name, 0) for vid_name in all_real_vid]
        vid_list += [(vid_name, 1) for vid_name in all_fake_vid]
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
        all_images += [(f, label) for f in frames]
    return all_images


class FFIWDatasets(Dataset):
    def __init__(self, cfg, mode='train', trans=None, split_ratio=1.0, save_faces=False):
        super(FFIWDatasets, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.trans = trans
        self.expand_ratio = self.cfg.DATAS.EXPAND_RATIO
        self.root = '/home/og/home/lqx/mnt_point/datasets/FFIW/FFIW10K-v1-release/'
        self.save_faces = save_faces

        self.mode_list = get_ffiw_mode_list(self.root, mode)
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
                box = np.load(_.replace(_[-4:], '.npy').replace('frames', 'retinaface'))[0]
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


class FFIWTest(Dataset):
    def __init__(self, mode='train', trans=None, split_ratio=1.0) -> None:
        super().__init__()
        self.root = '/home/og/home/lqx/mnt_point/datasets/FFIW/FFIW10K-v1-release/'
        self.mode_list = get_ffiw_mode_list(self.root, mode)
        self.mode_list = choice_vids(self.mode_list, split_ratio=split_ratio)
        
        
    def __len__(self):
        return len(self.mode_list)
    
    def __getitem__(self, index):
        vid_num = self.mode_list[index]
        frame_paths = glob.glob(os.path.join(self.root, vid_num[0].replace('.mp4', '_frames/*.jpg')))
        retina_faces_paths = [_.replace(_[-4:], '.npy').replace('frames', 'retinaface') for _ in frame_paths]
        label = int(vid_num[1])
        return frame_paths, retina_faces_paths, label
    

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

    source_dp = FFIWDatasets(cfg, mode='test', trans=trans, save_faces=True)
    source_dp.stat_images()

    data = source_dp[320]
    print('down')
    # data_loader = torch.utils.data.DataLoader(source_dp, batch_size=32, num_workers=8)
    # all_freq = []
    # for data in data_loader:
    #     _, freq, _ = data
    #     print(torch.max(freq))
    #     print('done')
        # all_freq.append(freq)

