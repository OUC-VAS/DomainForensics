import os
import cv2
import PIL.Image as I
import numpy as np
from datasets.ffpp import crop_retina_faces

import retinaface


image_root = '/home/og/home/lqx/datasets/FF++/original_sequences/youtube/c23/frames/393'
fake_root = '/home/og/home/lqx/datasets/FF++/manipulated_sequences/Deepfakes/c23/frames/739_865'
# image_root = '/home/og/home/lqx/datasets/Celeb-DF/Celeb-DF-v2/Celeb-real_frames/'
# fake_root = '/home/og/home/lqx/datasets/Celeb-DF/Celeb-DF-v2/Celeb-synthesis_frames/'

# fake
# vid = 'id16_id20_0012'
# fnum = '286'

# real
vid = 'id22_0006'
fnum = '203'

# img_name = 'id57_id51_0000/425.jpg'
# img_name = vid + '/' + fnum + '.jpg'
out_name = 'faceswap_fake_173.jpg'
# out_name = 'celeb_real_' + vid + '_' + fnum + '.jpg'


img_name = '173.jpg'
save_path = '/home/og/home/lqx/pics/'

# img_path = os.path.join(image_root, img_name)
img_path = os.path.join(fake_root, img_name)

box = np.load(img_path.replace(img_path[-4:], '.npy').replace('frames', 'retinaface'))
img = I.open(img_path)
img = crop_retina_faces(img, box, expand_ratio=1.3)
img = np.array(img).astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

out_path = os.path.join(save_path, out_name)
# out_path = os.path.join(save_path, '888_df_c23.jpg')
# img.save(out_path)
cv2.imwrite(out_path, img)

