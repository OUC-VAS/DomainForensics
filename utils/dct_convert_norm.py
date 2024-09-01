import math

from scipy import fftpack
import numpy as np
import cv2


def dct2(array):
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array


def _welford_update(existing_aggregate, new_value):
    (count, mean, M2) = existing_aggregate
    if count is None:
        count, mean, M2 = 0, np.zeros_like(new_value), np.zeros_like(new_value)

    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2

    return (count, mean, M2)


def _welford_finalize(existing_aggregate):
    count, mean, M2 = existing_aggregate
    mean, variance, sample_variance = (mean, M2/count, M2/(count - 1))
    if count < 2:
        return (float("nan"), float("nan"), float("nan"))
    else:
        return (mean, variance, sample_variance)


def welford(sample):
    """Calculates the mean, variance and sample variance along the first axis of an array.
    Taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    existing_aggregate = (None, None, None)
    for data in sample:
        existing_aggregate = _welford_update(existing_aggregate, data)

    # sample variance only for calculation
    return _welford_finalize(existing_aggregate)[:-1]

def log_scale(array, eps=1e-12):
    """Log scale the input array.
    """
    array = np.abs(array)
    array += eps  # no zero in log
    array = np.log(array)
    return array


def dct_ycbcr(y):
    h, w = y.shape
    for i in range(h//8):
        for j in range(w//8):
            # y[i*8:(i+1)*8, j*8:(j+1)*8] = dct2(y[i*8:(i+1)*8, j*8:(j+1)*8])
            y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]).astype(np.float64)
    return y


if __name__ == '__main__':
    from datasets.ff import FFDatasets
    from configs.defaults import get_config
    from torch.utils.data import ConcatDataset, DataLoader
    import torch
    from datasets.trans import get_transfroms
    cfg = get_config()
    import tqdm

    methods = ['Deepfakes', 'NeuralTextures', 'Face2Face', 'FaceSwap'] #

    trans = get_transfroms(cfg, mode='train')
    dst_train = FFDatasets(cfg, mode='train', method=methods, quality='c40', trans=None)
    dst_val = FFDatasets(cfg, mode='val', method=methods, quality='c40', trans=None)
    # dst_test = FFDatasets(cfg, mode='test', method=methods, quality='c40', trans=None)
    dst = ConcatDataset([dst_train]) #
    dloder = DataLoader(dst, batch_size=32, num_workers=4, shuffle=False)

    print(len(dst))
    import sys
    sys.exit(0)

    curmax = np.zeros((224, 224, 3))


    all_imgs = []
    for data in tqdm.tqdm(dloder):
        imgs, lbl = data
        imgs_dct = []
        for i in range(imgs.size()[0]):
            img = imgs[i].numpy()
            img_ycbcr_dct = np.float64(np.zeros_like(img))
            img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float64)
            Y, Cr, Cb = img_ycbcr[:,:, 0], img_ycbcr[:,:, 1], img_ycbcr[:,:, 2]
            img_ycbcr_dct[:, :, 0] = dct_ycbcr(Y)
            img_ycbcr_dct[:, :, 1] = dct_ycbcr(Cr)
            img_ycbcr_dct[:, :, 2] = dct_ycbcr(Cb)
            # reshape
            h,w,c = img_ycbcr_dct.shape
            img_ycbcr_dct = img_ycbcr_dct.reshape(h//8, 8, w//8, 8, c).transpose(0, 2, 1, 3, 4).reshape(h//8, w//8, 64, c).reshape(h//8, w//8, 64*c)
            imgs_dct.append(torch.from_numpy(img_ycbcr_dct).float())

            # img_dct = dct2(img)
            # img_dct = log_scale(img_dct)
            # max_values = np.absolute(img_dct)

            # mask = curmax > max_values
            # curmax *= mask
            # curmax += max_values * ~mask
            # imgs_dct[i] = torch.tensor(img_dct)
        imgs_dct_stack = torch.stack(imgs_dct, dim=0)
        all_imgs.append(imgs_dct_stack) # B, H, W, 192

    # compute mean and std
    all_imgs = torch.cat(all_imgs, dim=0)
    N, H, W, C = all_imgs.shape
    all_imgs = all_imgs.view(-1, C)
    mean = torch.mean(all_imgs, dim=0)
    std = torch.std(all_imgs, dim=0)

    # curmax = torch.tensor(curmax)
    # all_imgs = torch.cat(all_imgs, dim=0)
    # all_imgs = all_imgs / curmax.unsqueeze(0)
    # all_imgs = all_imgs.numpy()
    #
    # # compute mean and std
    # mean, var = welford(all_imgs)
    # std = np.sqrt(var)
    # mean, std = torch.tensor(mean), torch.tensor(std)

    freq_data = {
        'mean': mean,
        'std': std,
        # 'maxvals': curmax
    }
    torch.save(freq_data, './freq_dct_mean_val.pth')
    print('done')


