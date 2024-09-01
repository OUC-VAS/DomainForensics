# self-blending image transform
import torch
import torch.nn as nn
import numpy as np
import PIL.Image as I
import cv2
import albumentations as alb
import datasets.blend as B
import torchvision.transforms as tfm


class SBITransform(nn.Module):
    def __init__(self, quality='c40'):
        super(SBITransform, self).__init__()
        self.quality = quality
        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.image_size = (224, 224)

    def forward(self, img_path):
        img = np.array(I.open(img_path))

        landmark = self.load_landmark(img_path)
        bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()])
        bbox = np.load(img_path.replace(img_path[-4:], '.npy').replace('/frames/', '/retinaface/'))
        bbox[-2:] = bbox[:-2] + bbox[-2:]
        iou = IoUfrom2bboxes(bbox_lm, bbox)
        if iou < 0.4:
            bbox = bbox_lm

        landmark = self.reorder_landmark(landmark)

        # if np.random.rand() < 0.5:
        #     img, _, landmark, bbox = self.hflip(img, None, landmark, bbox)
        img, landmark, bbox, __ = crop_face(img, landmark, bbox, margin=True, crop_by_bbox=False)
        img_r, img_f, mask_f = self.self_blending(img.copy(), landmark.copy())
        imgs_trans = self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
        img_f = imgs_trans['image']
        img_r = imgs_trans['image1']
        img_f, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(img_f, landmark, bbox, margin=False,
                                                                      crop_by_bbox=True, abs_coord=True,
                                                                      phase='train')
        img_r = img_r[y0_new:y1_new, x0_new:x1_new]
        try:
            img_f = cv2.resize(img_f, self.image_size, interpolation=cv2.INTER_LINEAR)
            img_r = cv2.resize(img_r, self.image_size, interpolation=cv2.INTER_LINEAR)
        except:
            print("Debug ---- ", img_path)
        return img_r, img_f

    def self_blending(self, img, landmark):
        if np.random.rand() < 0.25:
            landmark = landmark[:68]
        mask = np.zeros_like(img[:, :, 0])
        cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        source = img.copy()
        if np.random.rand() < 0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source, mask)

        img_blended, mask = B.dynamic_blend(source, img, mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)
        return img, img_blended, mask

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                       val_shift_limit=(-0.3, 0.3), p=1),
                alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
            ], p=1),

            alb.OneOf([
                RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=1),

        ], p=1.)

    def get_transforms(self):
        return alb.Compose([

            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                   val_shift_limit=(-0.3, 0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
            alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),

        ],
            additional_targets={f'image1': 'image'},
            p=1.)

    def randaffine(self, img, mask):
        f = alb.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1)

        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']

        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask

    def hflip(self, img, mask=None, landmark=None, bbox=None):
        H, W = img.shape[:2]
        landmark = landmark.copy()
        bbox = bbox.copy()

        if landmark is not None:
            landmark_new = np.zeros_like(landmark)

            landmark_new[:17] = landmark[:17][::-1]
            landmark_new[17:27] = landmark[17:27][::-1]

            landmark_new[27:31] = landmark[27:31]
            landmark_new[31:36] = landmark[31:36][::-1]

            landmark_new[36:40] = landmark[42:46][::-1]
            landmark_new[40:42] = landmark[46:48][::-1]

            landmark_new[42:46] = landmark[36:40][::-1]
            landmark_new[46:48] = landmark[40:42][::-1]

            landmark_new[48:55] = landmark[48:55][::-1]
            landmark_new[55:60] = landmark[55:60][::-1]

            landmark_new[60:65] = landmark[60:65][::-1]
            landmark_new[65:68] = landmark[65:68][::-1]
            if len(landmark) == 68:
                pass
            elif len(landmark) == 81:
                landmark_new[68:81] = landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:, 0] = W - landmark_new[:, 0]

        else:
            landmark_new = None

        if bbox is not None:
            bbox_new = np.zeros_like(bbox)
            bbox_new[0, 0] = bbox[1, 0]
            bbox_new[1, 0] = bbox[0, 0]
            bbox_new[:, 0] = W - bbox_new[:, 0]
            bbox_new[:, 1] = bbox[:, 1].copy()
            if len(bbox) > 2:
                bbox_new[2, 0] = W - bbox[3, 0]
                bbox_new[2, 1] = bbox[3, 1]
                bbox_new[3, 0] = W - bbox[2, 0]
                bbox_new[3, 1] = bbox[2, 1]
                bbox_new[4, 0] = W - bbox[4, 0]
                bbox_new[4, 1] = bbox[4, 1]
                bbox_new[5, 0] = W - bbox[6, 0]
                bbox_new[5, 1] = bbox[6, 1]
                bbox_new[6, 0] = W - bbox[5, 0]
                bbox_new[6, 1] = bbox[5, 1]
        else:
            bbox_new = None

        if mask is not None:
            mask = mask[:, ::-1]
        else:
            mask = None
        img = img[:, ::-1].copy()
        return img, mask, landmark_new, bbox_new


    def reorder_landmark(self, landmark):
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark

    def load_landmark(self, img_path):
        if self.quality == 'c40':
            suffix = '.png'
        else:
            suffix = '.jpg'
        landmark = np.load(img_path.replace(suffix, '.npy').replace('/frames/', '/landmarks/'))[0]
        return landmark


def crop_face(img, landmark=None, bbox=None, margin=False, crop_by_bbox=True, abs_coord=False, only_img=False,
              phase='train'):
    assert phase in ['train', 'val', 'test']

    # crop face------------------------------------------
    H, W = len(img), len(img[0])

    assert landmark is not None or bbox is not None

    H, W = len(img), len(img[0])

    if crop_by_bbox:
        x0, y0, x1, y1 = bbox
        # x0, y0 = bbox[0]
        # x1, y1 = bbox[1]
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 4  # 0#np.random.rand()*(w/8)
        w1_margin = w / 4
        h0_margin = h / 4  # 0#np.random.rand()*(h/5)
        h1_margin = h / 4
    else:
        x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
        x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 8  # 0#np.random.rand()*(w/8)
        w1_margin = w / 8
        h0_margin = h / 2  # 0#np.random.rand()*(h/5)
        h1_margin = h / 5

    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == 'train':
        w0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
        w1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
        h0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
        h1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5

    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i, (p, q) in enumerate(landmark):
            landmark_cropped[i] = [p - x0_new, q - y0_new]
    else:
        landmark_cropped = None
    if bbox is not None:
        bbox_cropped = np.zeros_like(bbox)
        bbox_cropped[::2] = bbox[::2] - x0_new
        bbox_cropped[1::2] = bbox[1::2] - y0_new
        # for i, (p, q) in enumerate(bbox):
        #     bbox_cropped[i] = [p - x0_new, q - y0_new]
    else:
        bbox_cropped = None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped, landmark_cropped, bbox_cropped, (
        y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1), y0_new, y1_new, x0_new, x1_new
    else:
        return img_cropped, landmark_cropped, bbox_cropped, (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1)




def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self,img,**params):
        return self.randomdownscale(img)

    def randomdownscale(self,img):
        keep_ratio=True
        keep_input_shape=True
        H,W,C=img.shape
        ratio_list=[2,4]
        r=ratio_list[np.random.randint(len(ratio_list))]
        img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

        return img_ds


if __name__ == '__main__':
    img_path = '/home/og/home/lqx/datasets/FF++/original_sequences/youtube/c40/frames/000/000.png'

    sbi = SBITransform(quality='c40')

    result = sbi(img_path)

    img_r, img_f = result

