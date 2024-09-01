import torch
import PIL.Image as Image
from functools import partial
import cv2

import torchvision.transforms.functional as F
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop, PILToTensor, ConvertImageDtype,\
    Normalize, Resize, RandomAutocontrast, RandomRotation, CenterCrop, ColorJitter, RandomApply, RandomGrayscale, ToTensor
from torch import Tensor

from PIL import ImageFilter
import random
from datasets.randaug import RandAugment
import numpy as np
import torch.nn as nn

from utils.dct_convert_norm import dct2, log_scale
from utils.dct_convert_norm import dct_ycbcr
import albumentations as alb
import PIL.Image as I
import numpy as np
import cv2


class RandomCompression(nn.Module):
    def __init__(self, quality_low=99, quality_high=100, p=0.5):
        super(RandomCompression, self).__init__()
        self.comp = alb.ImageCompression(quality_lower=quality_low, quality_upper=quality_high, p=p)

    def forward(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = self.comp(image=img)['image']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = I.fromarray(img)
        return img
    
def trans_func(data, trans=None):
    # only for segmentation
    img, lbl = data
    if trans is not None:
        img = trans(img)
    return img, lbl


def trans_wrapper(dp, mode='train', cfg=None):
    # TODO: build transformation according to `cfg` and `mode`
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    # size = (256, 256)

    # xception mean and std
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    size = (299, 299)

    if mode == 'train':
        trans = Compose([
            Resize(size),
            RandomHorizontalFlip(0.5),
            RandomAutocontrast(p=0.5),
            RandomRotation(15),
            RandomCrop(size=size),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
    elif mode == 'val':
        trans = Compose([
            Resize(size),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(mean=mean, std=std),
        ])
    else:
        trans = Compose([
            Resize(size),
            PILToTensor(),
            ConvertImageDtype(torch.float),
            Normalize(mean=mean, std=std),
        ])
    # end ----

    ts_func = partial(trans_func, trans=trans)
    dp = dp.map(fn=ts_func)
    dp = dp.sharding_filter()
    return dp


def get_transfroms(cfg=None, mode='train'):

    mean = cfg.AUGS.MEAN
    std = cfg.AUGS.STD
    size = cfg.AUGS.CROP_SIZE

    if mode == 'train':
        trans_list = [
            Resize(size),
            RandomHorizontalFlip(cfg.AUGS.FLIP_PROB),
            RandomAutocontrast(p=cfg.AUGS.CONTRAST_PROB),
            RandomCompression(quality_low=40, quality_high=100, p=0.5)
        ]
        trans = Compose(trans_list)
    elif mode == 'val':
        trans_list = [
            Resize(size),
            RandomHorizontalFlip(cfg.AUGS.FLIP_PROB),
        ]
        trans = Compose(trans_list)

    else:
        trans_list = [
            Resize(size),
            RandomHorizontalFlip(cfg.AUGS.FLIP_PROB),
        ]
        trans = Compose(trans_list)
    return trans


def get_contrastive_trans(cfg=None):
    size = cfg.AUGS.CROP_SIZE
    mean = cfg.AUGS.MEAN
    std = cfg.AUGS.STD
    trans = []
    trans.extend([Resize(size), RandomHorizontalFlip(p=0.5)])
    color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
    trans.extend(color_transform)
    trans.extend(
        [ToTensor(),
         Normalize(mean, std)]
    )
    trans = Compose(trans)
    trans.transforms.insert(0, RandAugment(2, 14))
    return trans


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = RandomApply([color_jitter], p=0.8)
    rnd_gray = RandomGrayscale(p=0.2)
    color_distort = Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def _rgb_to_y(r: Tensor, g: Tensor, b: Tensor) -> Tensor:
    y: Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return y

def rgb_to_ycbcr(image: Tensor) -> Tensor:
    r"""Convert an RGB image to YCbCr.

    .. image:: _static/img/rgb_to_ycbcr.png

    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0, :, :]
    g: Tensor = image[..., 1, :, :]
    b: Tensor = image[..., 2, :, :]

    delta: float = 0.5
    y: Tensor = _rgb_to_y(r, g, b)
    cb: Tensor = (b - y) * 0.564 + delta
    cr: Tensor = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)


class RgbToYcbcr(nn.Module):
    r"""Convert an image from RGB to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    Returns:
        YCbCr version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_ycbcr(image)

