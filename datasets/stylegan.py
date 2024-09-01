import os
import glob
import torch
import numpy as np
import PIL.Image as I

from datasets.ff import get_model_list, get_face_images, crop_retina_faces, choice_vids, face_filter
import torchvision.transforms as tfm
from torch.utils.data import Dataset
from datasets.dct_transforms import get_dct_transform


def get_style_gan_images(root, mode='train'):
    split_file = os.path.join(root, mode+'.txt')
    with open(split_file) as f:
        data = f.readlines()
    data_list = []
    for _ in data:
        fname, label = _.split(' ')
        label = int(label.replace('\n', ''))
        data_list.append((os.path.join(root, fname), label))
    return data_list


class StyleGANDatasets(Dataset):
    def __init__(self, cfg, mode='train', trans=None, split_ratio=1.0, return_name=False):
        super(StyleGANDatasets, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.trans = trans
        self.face_data = get_style_gan_images(self.cfg.DATAS.ROOT_STYLEGAN, mode)
        print(self.face_data)
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
        self.return_name = return_name

    def __len__(self):
        return len(self.face_data)

    def __getitem__(self, idx):
        # Frame level
        fname, label = self.face_data[idx]
        # open iamge
        img = I.open(fname)
        if self.trans is not None:
            if self.freq_dct is None:
                img_trans = self.trans(img)
                img_trans = self.to_tensor(img_trans)
                if self.return_name:
                    return img_trans, label, fname
                else:
                    return img_trans, label
            else:
                img_trans = self.trans(img)
                img_freq = self.freq_dct(img_trans).float()
                img_trans = self.to_tensor(img_trans)
                if self.return_name:
                    return img_trans, img_freq, label, fname
                else:
                    return img_trans, img_freq, label
        else:
            # mode = torchvision.transforms.InterpolationMode.BILINEAR
            # img = torchvision.transforms.functional.resize(img, size=[224, 224], interpolation=mode)
            img = self.temp_trans(img)
            img = np.asarray(img)
            return img, label


if __name__ == '__main__':
    root = '/home/og/home/lqx/datasets/stylegan'

    data_list = get_style_gan_images(root, 'train')
    print(data_list)
    print('done')
    #
    from configs.defaults import get_config
    from datasets.trans import get_transfroms
    cfg = get_config()
    trans = get_transfroms(cfg, mode='train')
    file = '/home/og/home/lqx/code/FaceAdaptation/configs/yamls/final.yaml'
    cfg.merge_from_file(file)
    cfg.DATAS.WITH_FREQUENCY = True
    cfg.DATAS.ROOT = root

    dst = StyleGANDatasets(cfg=cfg, mode='train', trans=trans)

    for d in dst:
        x, freq, y = d
        print('ddd')


