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
from datasets.trans import DCTTransform, get_transfroms
from datasets.dct_transforms import get_dct_transform

import matplotlib.pyplot as plt

import time

class FFTestDataset(Dataset):
    def __init__(self, cfg, mode='train', method='Deepfakes', quality='c40', trans=None,
                 split_ratio=1.0):
        super(FFTestDataset, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.test_imagenum_per_vid = self.cfg.TESTING.IMAGENUM_PER_VID

        # get video list
        mode_list = get_test_video_list(self.cfg.DATAS.ROOT, methods=method, mode=mode)
        # mode_list.sort()
        self.mode_list = mode_list
        # (fname, lbl)

        self.length_ratio = 1
        crop_face_images = get_test_face_images(self.cfg.DATAS.ROOT, mode_list=mode_list, method=method,
                                           quality=quality, length_ratio=self.length_ratio)
        self.face_data = crop_face_images
        # transforms
        self.trans = trans
        self.expand_ratio = self.cfg.DATAS.EXPAND_RATIO
        self.mean = cfg.AUGS.MEAN
        self.std = cfg.AUGS.STD

        self.temp_trans = get_transfroms(cfg, mode='train')

        self.to_tensor = tfm.Compose([
            tfm.ToTensor(),
            tfm.Normalize(mean=self.mean, std=self.std)
        ])
        if self.cfg.DATAS.WITH_FREQUENCY:
            self.freq_dct = get_dct_transform(cfg)
        else:
            self.freq_dct = None

    def __len__(self):
        return len(self.mode_list)

    def __getitem__(self, idx):
        # Video level
        m, vid_num = self.mode_list[idx]
        
        fnames = np.array([fn for fn in self.face_data if vid_num == fn[0].split('/')[-2]])
        images = []
        imgs_freq = []
        label = int(fnames[0][1])
        for _ in fnames:
            box = np.load(_[0].replace(_[0][-4:], '.npy').replace('/frames/', '/retinaface/'))
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

class FFDatasets(Dataset):
    def __init__(self, cfg, mode='train', method='Deepfakes', quality='c40', trans=None,
                 split_ratio=1.0):
        super(FFDatasets, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.test_imagenum_per_vid = self.cfg.TESTING.IMAGENUM_PER_VID

        # get video list
        mode_list = get_model_list(self.cfg.DATAS.ROOT, mode)
        mode_list.sort()
        mode_list = choice_vids(mode_list, split_ratio)
        self.mode_list = mode_list
        # (fname, lbl)

        if self.cfg.DATAS.TARGET[0] == 'CelebDF':
            self.length_ratio = 1
        else:
            self.length_ratio = 1
        crop_face_images = get_face_images(self.cfg.DATAS.ROOT, mode_list=mode_list, method=method,
                                           quality=quality, length_ratio=self.length_ratio)
        self.face_data = crop_face_images
        # transforms
        self.trans = trans
        self.expand_ratio = self.cfg.DATAS.EXPAND_RATIO
        self.mean = cfg.AUGS.MEAN
        self.std = cfg.AUGS.STD

        self.temp_trans = get_transfroms(cfg, mode='train')

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
            fnames = np.array([fn for fn in self.face_data if vid_num == fn[0].split('/')[-2]])
            images = []
            imgs_freq = []
            label = int(fnames[0][1])
            for _ in fnames:
                box = np.load(_[0].replace(_[0][-4:], '.npy').replace('/frames/', '/retinaface/'))
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
            box = np.load(fname.replace(fname[-4:], '.npy').replace('/frames/', '/retinaface/'))
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


def get_face_images(root, mode_list, method='Deepfakes', quality='c40', length_ratio=1):
    if not isinstance(method, list):
        method = [method]
    if quality == 'c40':
        suffix = 'png'
    elif quality == 'c23':
        suffix = 'jpg'
    else:
        suffix = 'png'
    all_real_images = glob.glob(os.path.join(root, 'original_sequences', 'youtube', quality, 'frames/*/*.' + suffix))
    all_real_images = [(fname, 0) for fname in all_real_images if mode_filter(fname, mode_list)] * length_ratio
    all_real_images = all_real_images # * len(method)
    all_fake_images = []
    if len(method) == 1:
        ratio = 1
    else:
        ratio = 1 / len(method) * length_ratio

    for m in method:
        assert m in ['Deepfakes', 'NeuralTextures', 'Face2Face', 'FaceSwap']
        fake_images = glob.glob(os.path.join(root, 'manipulated_sequences', m, quality, 'frames/*/*.' + suffix))
        fake_images = [fname for fname in fake_images if mode_filter(fname, mode_list)]
        # fake_images = choice_frames(fake_images, ratio)

        if len(method) == 1:
            fake_images = [(fname, 1) for fname in fake_images] * length_ratio
        else:
            fake_images = [(fname, 1) for fname in fake_images]

        all_fake_images += fake_images
    all_images = all_real_images + all_fake_images
    all_images = [item for item in all_images if face_filter(item[0])]
    return all_images# * 3

def get_test_face_images(root, mode_list, method='Deepfakes', quality='c40', length_ratio=1):
    if not isinstance(method, list):
        method = [method]
    if quality == 'c40':
        suffix = 'png'
    elif quality == 'c23':
        suffix = 'jpg'
    else:
        suffix = 'png'
    
    def real_filter(fname, mode_list):
        vid_num = fname.split('/')[-2]
        vid_num = vid_num[:3]
        return ('real', vid_num) in mode_list
    
    def method_filter(fname, mode_list, method):
        vid_num = fname.split('/')[-2]
        return (method, vid_num) in mode_list
    
    all_real_images = glob.glob(os.path.join(root, 'original_sequences', 'youtube', quality, 'frames/*/*.' + suffix))
    all_real_images = [(fname, 0) for fname in all_real_images if real_filter(fname, mode_list)] * length_ratio
    all_real_images = all_real_images # * len(method)
    all_fake_images = []

    ratio = 1

    for m in method:
        assert m in ['Deepfakes', 'NeuralTextures', 'Face2Face', 'FaceSwap']
        fake_images = glob.glob(os.path.join(root, 'manipulated_sequences', m, quality, 'frames/*/*.' + suffix))
        fake_images = [fname for fname in fake_images if method_filter(fname, mode_list, m)]
        # fake_images = choice_frames(fake_images, ratio)

        if len(method) == 1:
            fake_images = [(fname, 1) for fname in fake_images] * length_ratio
        else:
            fake_images = [(fname, 1) for fname in fake_images]

        all_fake_images += fake_images
    all_images = all_real_images + all_fake_images
    all_images = [item for item in all_images if face_filter(item[0])]
    return all_images# * 3


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

def get_test_video_list(root, methods, mode):
    json_file = os.path.join(root, 'splits', mode + '.json')
    mode_list = []
    json_data = json.load(open(json_file, 'r'))

    for d in json_data:
        mode_list += [('real', d[0]), ('real', d[1])]

    if mode == 'test':
        for m in methods:
            for d in json_data:
                fake_vid = [(m, d[0]+'_'+d[1]), (m, d[1]+'_'+d[0])]
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


def landmark_filter(fname):
    landmark_file = fname.replace('.png', '.npy').replace('/frames/', '/landmarks/')
    return os.path.exists(landmark_file)


def crop_faces(data, mode='train'):
    img, lbl, fname = data
    W, H = img.size

    landmark = np.load(fname.replace('.png', '.npy').replace('/frames/', '/landmarks/'))[0]
    # bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()])
    # bboxes = np.load(fname.replace('.png', '.npy').replace('/frames/', '/retina/'))
    # iou_max = -1
    # for i in range(len(bboxes)):
    #     iou = IoUfrom2bboxes(bbox_lm, bboxes[i].flatten())
    #     if iou_max < iou:
    #         bbox = bboxes[i]
    #         iou_max = iou

    x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
    x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
    w = x1 - x0
    h = y1 - y0
    w0_margin = w / 8  # 0#np.random.rand()*(w/8)
    w1_margin = w / 8
    h0_margin = h / 2  # 0#np.random.rand()*(h/5)
    h1_margin = h / 5

    if mode == 'train':
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5
    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)
    face = img.crop((x0_new, y0_new, x1_new, y1_new))
    return face, lbl


# def build_ffpp_datapipe(root, mode='train', method='Deepfakes', quality='c40'):
#     mode_list = get_model_list(root, mode)
#     mode_list.sort()
#     mode_func = partial(mode_filter, mode_list=mode_list)
#     method_func = partial(method_filter, method=method)
#     quality_func = partial(quality_filter, quality=quality)
#     crop_func = partial(crop_faces, mode=mode)
#
#     dp = FileLister(root=root, recursive=True, masks='*.png')
#     dp = dp.filter(filter_fn=mode_func)
#     dp = dp.filter(filter_fn=method_func)
#     dp = dp.filter(filter_fn=quality_func)
#     dp = dp.filter(filter_fn=landmark_filter)
#     print("Dataset Length is ", len(list(iter(dp))))
#     dp = dp.shuffle()
#     dp = dp.map(fn=label_mapper)
#     dp = dp.map(fn=PILopen)
#     dp = dp.map(fn=crop_func)
#     return dp

if __name__ == '__main__':
    from configs.defaults import get_config
    from datasets.trans import get_transfroms
    cfg = get_config()
    print(cfg.DATAS.WITH_FREQUENCY)
    trans = get_transfroms(cfg, mode='train')
    config_file = '/home/og/home/lqx/code/FaceAdaptation/configs/yamls/ssrt.yaml'
    cfg.merge_from_file(config_file)
    cfg.DATAS.WITH_FREQUENCY = True
    cfg.DATAS.SOURCE = ['Deepfakes', 'Face2Face', 'NeuralTextures', 'FaceSwap'] #'NeuralTextures'
    # cfg.DATAS.TARGET = ['Deepfakes', 'Face2Face', 'FaceSwap']
    cfg.DATAS.TARGET = ['FaceSwap']

    source_dp = FFDatasets(cfg, mode='train', method=cfg.DATAS.SOURCE, quality='c40',
                           trans=trans)
    source_dp.stat_images()


