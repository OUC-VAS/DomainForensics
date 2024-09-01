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


def get_celeb_mode_list(root, mode='train'):
    vid_list = []
    phrases = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
    with open(os.path.join(root, 'List_of_testing_videos.txt'), 'r') as f:
        test_vids = f.readlines()
    test_vids = [_.split('/')[1].replace('\n', '')[:-4] for _ in test_vids]

    for p in phrases:
        vid_path = os.path.join(root, p, '*.mp4')
        vids = glob.glob(vid_path)
        # filter by mode
        if mode == 'train':
            vids = [_.split('/')[-1][:-4] for _ in vids if os.path.basename(_)[:-4] not in test_vids]
        else:
            vids = [_.split('/')[-1][:-4] for _ in vids if os.path.basename(_)[:-4] in test_vids]
        vid_list += vids
    return vid_list


def face_celeb_filter(fname):
    face_file = fname.replace(fname[-4:], '.npy').replace('frames', 'retinaface')
    if os.path.exists(face_file):
        return True
    else:
        return False


def get_celeb_face_images(root, mode_list, mode='train'):
    suffix = 'jpg'
    phrases = ['Celeb-real', 'YouTube-real'] #
    all_images = []
    for p in phrases:
        all_p_images = glob.glob(os.path.join(root, p +'_frames/*/*.' + suffix))
        all_p_images = [_ for _ in all_p_images if _.split('/')[-2] in mode_list]
        all_images += all_p_images

    all_fake_images = glob.glob(os.path.join(root, 'Celeb-synthesis_frames/*/*.' + suffix))
    all_fake_images = [_ for _ in all_fake_images if _.split('/')[-2] in mode_list]

    all_images += all_fake_images
    all_images = [(_, 0) if 'real' in _ else (_, 1) for _ in all_images]
    all_images = [item for item in all_images if face_celeb_filter(item[0])]
    return all_images


class CelebDatasets(Dataset):
    def __init__(self, cfg, mode='train', trans=None, split_ratio=1.0):
        super(CelebDatasets, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.trans = trans
        self.expand_ratio = self.cfg.DATAS.EXPAND_RATIO
        self.mode_list = get_celeb_mode_list(self.cfg.DATAS.ROOT_CELEB, mode)
        self.mode_list = choice_vids(self.mode_list, split_ratio=split_ratio)
        self.face_data = get_celeb_face_images(self.cfg.DATAS.ROOT_CELEB, self.mode_list, mode=mode)
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
        real_num = len([_ for _ in self.face_data if 'real' in _[0]])
        fake_num = len([_ for _ in self.face_data if 'synthesis' in _[0]])
        print("Total real num is : ", real_num, ", Total fake num is : ", fake_num)

    def __getitem__(self, idx):
        if self.mode == 'test':
            # Video level
            vid_num = self.mode_list[idx]
            fnames = np.array([fn for fn in self.face_data if vid_num == fn[0].split('/')[-2]])
            images = []
            imgs_freq = []
            label = int(fnames[0][1])
            for _ in fnames:
                box = np.load(_[0].replace(_[0][-4:], '.npy').replace('frames', 'retinaface'))
                img = I.open(_[0])
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
                # mode = torchvision.transforms.InterpolationMode.BILINEAR
                # img = torchvision.transforms.functional.resize(img, size=[224, 224], interpolation=mode)
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

    source_dp = CelebDatasets(cfg, mode='train', trans=trans)
    source_dp.stat_images()
    # data_loader = torch.utils.data.DataLoader(source_dp, batch_size=32, num_workers=8)
    # all_freq = []
    # for data in data_loader:
    #     _, freq, _ = data
    #     print(torch.max(freq))
    #     print('done')
        # all_freq.append(freq)

