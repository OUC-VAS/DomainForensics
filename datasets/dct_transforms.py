import cv2
import numpy as np
import random
import torch
import PIL
from turbojpeg import TurboJPEG
from jpeg2dct.numpy import loads

# ref : https://github.com/calmevtime/DCTNet
train_upscaled_static_dct_direct_mean_interp = \
    [-8.8182e+01, -6.1055e-03, -5.5242e-03,  2.0941e-03,  2.5734e-02,
         3.4716e-03,  7.9595e-03,  9.8994e-03,  4.4540e-01, -1.4832e-02,
        -5.4096e-03, -7.4766e-03, -1.6562e-03, -1.9316e-03, -9.0320e-04,
        -6.1158e-04, -2.9818e-02, -6.7268e-03, -4.6165e-03, -3.3243e-03,
         1.8651e-04, -9.9512e-04,  2.4505e-05, -9.2646e-04,  2.8136e-02,
        -6.9496e-03, -4.4221e-03, -3.6631e-03, -8.3106e-04, -1.3757e-04,
        -9.1237e-05,  4.6594e-04,  2.2675e-02,  1.0841e-02,  1.0454e-02,
         1.1511e-02,  6.9423e-04,  1.1407e-02,  1.1049e-02,  1.0841e-02,
         6.1502e-03, -3.9330e-03, -1.7815e-03, -8.6346e-04, -7.1301e-04,
        -1.2488e-04,  2.9626e-04,  5.2058e-04, -5.7681e-03, -1.1062e-03,
        -5.3184e-04,  5.1728e-05, -1.2414e-04, -3.3520e-04,  1.0278e-04,
        -2.5262e-04, -2.6499e-03, -1.5367e-04, -4.7991e-04, -1.6512e-04,
         1.8415e-04,  1.3256e-03,  5.1608e-04,  2.4008e-03, -5.8851e+01,
        -2.9627e-02,  1.9023e-02, -8.5525e-03,  1.0984e-02, -3.9767e-03,
         2.6271e-03, -1.4720e-02,  2.8950e-01, -2.3654e-03, -1.1576e-03,
        -2.5729e-03, -7.5419e-04,  4.5031e-04, -8.6147e-05, -8.1984e-04,
         1.4292e-02, -1.0066e-03, -2.6600e-04, -3.1656e-04, -1.2277e-04,
         6.2251e-07,  6.3507e-07, -9.8801e-05,  2.7193e-02, -3.1811e-03,
        -4.9865e-04, -2.8683e-03, -1.0999e-03,  8.9669e-04, -4.0730e-05,
        -1.2923e-03,  2.4890e-03, -5.3078e-04,  3.1278e-05, -8.7565e-04,
        -2.3156e-04,  4.9525e-04,  2.7904e-05, -4.6859e-04,  1.0699e-02,
         5.0242e-04, -1.8507e-04,  7.9123e-04,  3.3808e-04, -3.4007e-04,
         5.3228e-05,  4.9216e-04,  9.0492e-04, -1.7397e-05,  1.1708e-05,
        -3.8139e-05,  1.3488e-04,  1.6828e-05,  1.3683e-04, -5.9816e-06,
         2.7926e-03, -1.3629e-03, -1.8664e-04, -1.5320e-03, -3.2609e-04,
         3.8309e-04,  4.8488e-05, -5.4750e-04,  3.2445e+01, -3.6239e-02,
        -3.3916e-03, -8.8930e-03,  4.2545e-03, -3.4243e-03,  1.9204e-04,
        -1.2630e-02, -2.7221e-01, -2.7304e-03, -6.2416e-04, -1.8734e-03,
        -7.4740e-04,  4.9801e-04,  1.1181e-05, -8.6663e-04, -4.9801e-03,
        -6.4732e-04, -4.0511e-04, -1.7768e-04, -8.3338e-05, -3.4283e-05,
        -1.2094e-04, -1.4036e-04, -2.8937e-02, -2.5002e-03, -3.3898e-04,
        -3.2817e-03, -8.8626e-04,  9.8814e-04, -3.3790e-05, -1.3716e-03,
        -2.7065e-03, -6.5809e-04,  5.1675e-04, -9.5561e-04, -4.1538e-04,
         4.9981e-04,  2.6319e-04, -2.3596e-04, -6.3968e-03,  3.4680e-04,
        -1.8243e-05,  6.9165e-04,  2.8074e-04, -1.2950e-04, -4.3346e-05,
         3.0856e-04,  6.5064e-05, -8.6827e-05, -2.1444e-04, -1.9143e-05,
         7.3807e-05, -1.0663e-04,  2.3619e-05, -7.7510e-05, -3.6079e-03,
        -1.0953e-03, -2.3643e-04, -1.4278e-03, -2.7461e-04,  4.0152e-04,
         6.3488e-05, -5.1409e-04]

train_upscaled_static_dct_direct_std_interp = \
    [2.5855e+02, 2.9071e+00, 2.0845e+00, 1.4130e+00, 1.1701e+00, 8.6759e-01,
        6.9122e-01, 5.8060e-01, 4.1398e+00, 5.8617e-01, 6.1470e-01, 4.5447e-01,
        4.0050e-01, 3.1809e-01, 2.7612e-01, 2.2761e-01, 2.4944e+00, 6.1390e-01,
        6.6717e-01, 4.7776e-01, 4.2423e-01, 3.2902e-01, 2.8491e-01, 2.1966e-01,
        1.6784e+00, 4.6153e-01, 4.8442e-01, 3.8392e-01, 3.5155e-01, 2.8021e-01,
        2.4507e-01, 2.0886e-01, 1.3292e+00, 4.0824e-01, 4.3592e-01, 3.5314e-01,
        3.4084e-01, 2.6742e-01, 2.3533e-01, 1.8724e-01, 1.0298e+00, 3.2564e-01,
        3.4105e-01, 2.8464e-01, 2.7273e-01, 2.2903e-01, 1.9819e-01, 1.8741e-01,
        8.4280e-01, 2.8582e-01, 2.9759e-01, 2.5096e-01, 2.4122e-01, 1.9889e-01,
        1.8369e-01, 1.4845e-01, 7.3536e-01, 2.3464e-01, 2.2991e-01, 2.1014e-01,
        1.8973e-01, 1.8939e-01, 1.4978e-01, 2.2849e-01, 8.1391e+01, 1.1005e+00,
        8.5148e-01, 5.3843e-01, 3.6663e-01, 2.8946e-01, 2.2699e-01, 2.2429e-01,
        1.7142e+00, 2.8104e-01, 3.0232e-01, 2.0937e-01, 1.6913e-01, 1.3149e-01,
        1.1430e-01, 9.6635e-02, 1.0087e+00, 3.0271e-01, 3.2368e-01, 2.1083e-01,
        1.7507e-01, 1.3554e-01, 1.1540e-01, 9.6918e-02, 6.3892e-01, 2.1461e-01,
        2.1525e-01, 1.5991e-01, 1.3647e-01, 1.0983e-01, 9.5740e-02, 8.6749e-02,
        4.3794e-01, 1.7593e-01, 1.8201e-01, 1.3862e-01, 1.2552e-01, 1.0209e-01,
        9.0384e-02, 8.0904e-02, 3.4054e-01, 1.3764e-01, 1.4266e-01, 1.1260e-01,
        1.0384e-01, 8.8304e-02, 8.1405e-02, 7.3104e-02, 2.6921e-01, 1.2014e-01,
        1.2216e-01, 9.9281e-02, 9.2747e-02, 8.2366e-02, 7.6270e-02, 6.9455e-02,
        2.2662e-01, 1.0409e-01, 1.0391e-01, 9.0539e-02, 8.4251e-02, 7.4573e-02,
        7.0277e-02, 6.6828e-02, 7.6139e+01, 1.0520e+00, 8.2037e-01, 5.1660e-01,
        3.4587e-01, 2.7396e-01, 2.1269e-01, 2.1135e-01, 1.5490e+00, 2.6418e-01,
        2.8330e-01, 1.9472e-01, 1.5550e-01, 1.2061e-01, 1.0404e-01, 8.7923e-02,
        9.5960e-01, 2.8236e-01, 3.0302e-01, 1.9655e-01, 1.6104e-01, 1.2435e-01,
        1.0432e-01, 8.7528e-02, 6.0192e-01, 1.9881e-01, 2.0061e-01, 1.4838e-01,
        1.2481e-01, 9.9987e-02, 8.5772e-02, 7.7785e-02, 4.1468e-01, 1.6243e-01,
        1.6768e-01, 1.2673e-01, 1.1335e-01, 9.1606e-02, 7.9449e-02, 7.1900e-02,
        3.1773e-01, 1.2714e-01, 1.3094e-01, 1.0249e-01, 9.3069e-02, 7.8847e-02,
        7.1933e-02, 6.4993e-02, 2.5160e-01, 1.0878e-01, 1.1081e-01, 8.9177e-02,
        8.2305e-02, 7.3128e-02, 6.6894e-02, 6.1423e-02, 2.0994e-01, 9.4814e-02,
        9.4280e-02, 8.1887e-02, 7.5140e-02, 6.6646e-02, 6.2288e-02, 5.9860e-02]



train_upscaled_static_mean = [-8.8397e+01,  1.4242e-02,  3.4117e-03,  7.0792e-03,  2.4295e-02,
         4.8884e-03,  1.1029e-02,  1.4621e-02,  4.4380e-01, -1.4657e-02,
        -5.4878e-03, -7.4716e-03, -1.7573e-03, -1.9678e-03, -1.0014e-03,
        -6.7293e-04, -3.2971e-02, -6.9187e-03, -4.5815e-03, -3.2968e-03,
         2.9935e-04, -1.0084e-03,  9.2506e-05, -9.1809e-04,  2.7731e-02,
        -6.8864e-03, -4.3638e-03, -3.4875e-03, -7.4995e-04, -1.2685e-04,
        -9.9695e-05,  4.2945e-04,  2.1127e-02,  1.3356e-02,  1.2893e-02,
         1.3848e-02,  8.0353e-04,  1.3728e-02,  1.3314e-02,  1.3362e-02,
         5.6187e-03, -3.8503e-03, -1.6255e-03, -8.9914e-04, -6.0506e-04,
        -9.6180e-05,  2.8288e-04,  5.1148e-04, -6.0038e-03, -1.0580e-03,
        -5.0092e-04,  4.1829e-05, -1.0169e-04, -3.4269e-04,  1.6878e-04,
        -2.4374e-04, -6.3966e-04, -1.7590e-04, -5.0322e-04, -1.0794e-04,
         1.0911e-04,  1.2639e-03,  4.8865e-04,  2.3114e-03, -5.8709e+01,
        -1.4975e-01,  1.5083e-02, -1.5743e-01,  1.3664e-02, -2.1918e-01,
         1.7582e-02, -6.2055e-01,  1.3488e-01, -4.2079e-03, -1.0616e-03,
         8.0632e-05,  1.8554e-05,  4.6599e-04, -1.6110e-05,  3.7761e-04,
        -3.2042e-03, -2.1745e-03,  1.7846e-04, -2.5755e-03, -6.3193e-05,
        -6.1709e-03,  2.2338e-05, -2.7234e-02,  1.4375e-02, -1.5920e-04,
         1.0560e-04,  2.0734e-05,  8.9964e-06,  6.6089e-05, -2.2969e-05,
         8.5885e-05,  1.4262e-02,  1.3967e-02,  1.2928e-02,  1.1558e-02,
         5.9293e-03,  8.0587e-03,  1.2185e-02, -3.9466e-03,  3.9497e-03,
         3.2286e-04,  4.7901e-05,  1.1759e-05, -5.8365e-06,  3.3193e-06,
         2.2425e-05,  4.2256e-05,  5.2415e-04,  4.1421e-04, -9.4003e-05,
        -8.9865e-04,  5.5458e-05, -2.3636e-03,  7.7944e-05, -7.6696e-03,
         1.7445e-04,  8.6328e-05,  4.4323e-05,  1.4780e-05,  2.5238e-05,
        -9.2798e-06, -4.1784e-06, -1.3018e-05,  3.2116e+01, -1.4412e-01,
         1.2337e-02, -1.5120e-01,  8.5720e-03, -2.1042e-01,  1.5230e-02,
        -5.9077e-01, -1.3207e-01, -4.0276e-03, -9.3269e-04,  1.0862e-04,
         4.1762e-05,  4.6122e-04,  2.8299e-05,  3.2597e-04, -3.9396e-03,
        -2.2667e-03,  6.7284e-05, -2.6205e-03,  1.6429e-04, -6.6546e-03,
         2.5047e-04, -2.8640e-02, -1.1487e-02, -2.0323e-04,  2.9561e-05,
         3.4320e-05, -2.2354e-05,  5.4670e-05,  9.8016e-07,  9.6956e-05,
         9.3471e-03,  1.3498e-02,  1.2898e-02,  1.0983e-02,  5.5875e-03,
         7.5598e-03,  1.1768e-02, -5.1549e-03, -2.8011e-03,  2.3734e-04,
         3.9085e-05,  3.7114e-05,  3.3999e-05, -7.5000e-06, -1.0987e-05,
         7.5305e-05, -1.3658e-03,  3.0618e-04,  9.6108e-05, -9.3444e-04,
         4.5678e-05, -2.4390e-03,  1.2077e-04, -8.3779e-03, -3.4181e-04,
         1.3019e-04,  3.5683e-06,  5.3878e-06, -4.0862e-07, -9.4194e-06,
        -9.6086e-07,  2.7884e-05]

train_upscaled_static_std = [2.5996e+02, 2.7776e+00, 2.0454e+00, 1.3947e+00, 1.1929e+00, 8.6391e-01,
        6.8394e-01, 5.7572e-01, 3.9907e+00, 5.6620e-01, 5.9385e-01, 4.3958e-01,
        3.8698e-01, 3.0748e-01, 2.6694e-01, 2.2055e-01, 2.4791e+00, 5.9187e-01,
        6.4341e-01, 4.6003e-01, 4.0897e-01, 3.1736e-01, 2.7511e-01, 2.1240e-01,
        1.6547e+00, 4.4618e-01, 4.6693e-01, 3.6985e-01, 3.3952e-01, 2.6992e-01,
        2.3661e-01, 2.0211e-01, 1.3388e+00, 3.9514e-01, 4.2135e-01, 3.4065e-01,
        3.2892e-01, 2.5796e-01, 2.2765e-01, 1.8097e-01, 1.0203e+00, 3.1577e-01,
        3.2870e-01, 2.7429e-01, 2.6293e-01, 2.2137e-01, 1.9189e-01, 1.8148e-01,
        8.4185e-01, 2.7672e-01, 2.8746e-01, 2.4236e-01, 2.3329e-01, 1.9236e-01,
        1.7785e-01, 1.4372e-01, 7.3554e-01, 2.2719e-01, 2.2219e-01, 2.0297e-01,
        1.8347e-01, 1.8323e-01, 1.4515e-01, 2.2195e-01, 8.1161e+01, 4.9091e-01,
        3.1735e-01, 2.0449e-01, 1.1882e-01, 1.2908e-01, 5.0844e-02, 2.6862e-01,
        7.8728e-01, 1.0321e-01, 9.6477e-02, 6.2715e-02, 4.6128e-02, 3.3353e-02,
        2.3952e-02, 1.8714e-02, 3.8195e-01, 9.7172e-02, 1.0334e-01, 6.3937e-02,
        4.9453e-02, 3.5066e-02, 2.5177e-02, 3.2152e-02, 2.3052e-01, 6.3559e-02,
        6.5152e-02, 4.3934e-02, 3.5844e-02, 2.5748e-02, 1.9563e-02, 1.5570e-02,
        1.3944e-01, 4.9192e-02, 5.2346e-02, 3.7090e-02, 3.3560e-02, 2.3967e-02,
        1.8798e-02, 2.1980e-02, 9.8602e-02, 3.5452e-02, 3.6885e-02, 2.6706e-02,
        2.3650e-02, 1.7736e-02, 1.4376e-02, 1.2149e-02, 6.4540e-02, 2.6415e-02,
        2.7738e-02, 2.1018e-02, 1.9090e-02, 1.5044e-02, 1.2555e-02, 1.4048e-02,
        5.1324e-02, 1.9759e-02, 2.0474e-02, 1.6161e-02, 1.4880e-02, 1.2156e-02,
        1.0401e-02, 1.0198e-02, 7.5848e+01, 4.6409e-01, 2.9777e-01, 1.9486e-01,
        1.0790e-01, 1.2831e-01, 4.5828e-02, 2.7543e-01, 7.0298e-01, 9.4832e-02,
        8.7310e-02, 5.6864e-02, 4.0744e-02, 2.9727e-02, 2.1350e-02, 1.6915e-02,
        3.5970e-01, 8.7856e-02, 9.2837e-02, 5.6967e-02, 4.3606e-02, 3.1381e-02,
        2.2382e-02, 3.1701e-02, 2.1259e-01, 5.7565e-02, 5.8307e-02, 3.9093e-02,
        3.1663e-02, 2.2917e-02, 1.7423e-02, 1.4101e-02, 1.2614e-01, 4.3897e-02,
        4.6633e-02, 3.3022e-02, 2.9742e-02, 2.1378e-02, 1.6930e-02, 2.0954e-02,
        9.0127e-02, 3.1778e-02, 3.2939e-02, 2.3828e-02, 2.1016e-02, 1.5920e-02,
        1.2976e-02, 1.1180e-02, 5.8509e-02, 2.3753e-02, 2.4662e-02, 1.8851e-02,
        1.7072e-02, 1.3647e-02, 1.1419e-02, 1.3227e-02, 4.7294e-02, 1.7943e-02,
        1.8471e-02, 1.4692e-02, 1.3447e-02, 1.1150e-02, 9.6059e-03, 9.6861e-03]


INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def upscale(img, upscale_factor=None, desize_size=None, interpolation='BILINEAR'):
    h, w, c = img.shape
    if upscale_factor is not None:
        dh, dw = upscale_factor*h, upscale_factor*w
    elif desize_size is not None:
        # dh, dw = desize_size.shape
        dh, dw = desize_size
    else:
        raise ValueError
    return cv2.resize(img, dsize=(dw, dh), interpolation=INTER_MODE[interpolation])


def hflip(img):
    """Horizontally flip the given PIL Image.

    Args:
        img (np.ndarray): Image to be flipped.

    Returns:
        np.ndarray:  Horizontall flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    return cv2.flip(img, 1)


def to_tensor_dct(img):
    """Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W)

    Args:
        pic (np.ndarray, torch.Tensor): Image to be converted to tensor, (H x W x C[RGB]).

    Returns:
        Tensor: Converted image.
    """

    img = torch.from_numpy(img.transpose((2, 0, 1))).float()
    return img


def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See ``Normalize`` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if _is_tensor_image(tensor):
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor
    elif _is_numpy_image(tensor):
        return (tensor.astype(np.float32) - 255.0 * np.array(mean))/np.array(std)
    else:
        raise RuntimeError('Undefined type')



class UpsampleCbCr(object):
    def __init__(self, upscale_factor=2, interpolation='BILINEAR'):
        self.upscale_factor = upscale_factor
        self.interpolation = interpolation

    def __call__(self, img):
        y, cb, cr = img[0], img[1], img[2]

        dh, dw, _ = y.shape
        # y  = F.upscale(y,  desize_size=(dh, dw), interpolation=self.interpolation)
        cb = upscale(cb, desize_size=(dh, dw), interpolation=self.interpolation)
        cr = upscale(cr, desize_size=(dh, dw), interpolation=self.interpolation)

        return y, cb, cr


class Upscale(object):
    def __init__(self, upscale_factor=2, interpolation='BILINEAR'):
        self.upscale_factor = upscale_factor
        self.interpolation = interpolation

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        return img, upscale(img, self.upscale_factor, self.interpolation)


class Aggregate2(object):
    def __call__(self, img):
        dct_y, dct_cb, dct_cr = img[0], img[1], img[2]
        try:
            dct_y = np.concatenate((dct_y, dct_cb, dct_cr), axis=2)
        except:
            print('Y: {}, Cb: {}, Cr: {}'.format(dct_y.shape, dct_cb.shape, dct_cr.shape))
        return dct_y

class Aggregate(object):
    def __call__(self, img):
        dct_y, dct_cb, dct_cr = img[0], img[1], img[2]
        dct_y = torch.cat((dct_y, dct_cb, dct_cr), dim=0)
        return dct_y


class RandomHorizontalFlip(object):
    """Horizontally flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be flipped.

        Returns:
            CV Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ToTensorDCT2(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor_dct(img)


class NormalizeDCT(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, y_mean, y_std, cb_mean=None, cb_std=None, cr_mean=None, cr_std=None, channels=None, pattern='square'):
        self.y_mean,  self.y_std = y_mean, y_std
        self.cb_mean, self.cb_std = cb_mean, cb_std
        self.cr_mean, self.cr_std = cr_mean, cr_std

        self.mean_y, self.std_y = y_mean, y_std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(tensor, list):
            y, cb, cr = tensor[0], tensor[1], tensor[2]
            y  = normalize(y,  self.y_mean,  self.y_std)
            cb = normalize(cb, self.cb_mean, self.cb_std)
            cr = normalize(cr, self.cr_mean, self.cr_std)
            return y, cb, cr
        else:
            y = normalize(tensor, self.mean_y, self.std_y)
            return y


class ToTensorDCT(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        y, cb, cr = img[0], img[1], img[2]
        y, cb, cr = to_tensor_dct(y), to_tensor_dct(cb), to_tensor_dct(cr)

        return y, cb, cr

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def transform_dct(img, encoder):
    if img.dtype != 'uint8':
        img = np.ascontiguousarray(img, dtype="uint8")
    img = encoder.encode(img, quality=100, jpeg_subsample=2)
    dct_y, dct_cb, dct_cr = loads(img)  # 28
    return dct_y, dct_cb, dct_cr


class TransformUpscaledDCT(object):
    def __init__(self):
        self.jpeg_encoder = TurboJPEG('/usr/lib/libturbojpeg.so')

    def __call__(self, img):
        y, cbcr = img[0], img[1]
        dct_y, _, _ = transform_dct(y, self.jpeg_encoder)
        _, dct_cb, dct_cr = transform_dct(cbcr, self.jpeg_encoder)
        return dct_y, dct_cb, dct_cr


def get_dct_transform(cfg=None):
    trans = Compose([
        Upscale(upscale_factor=2),
        TransformUpscaledDCT(),
        ToTensorDCT(),
        Aggregate(),
        NormalizeDCT(train_upscaled_static_mean, train_upscaled_static_std)
    ])
    return trans




