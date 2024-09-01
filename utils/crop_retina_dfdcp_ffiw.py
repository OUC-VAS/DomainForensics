from glob import glob
import os
import sys
sys.path.append('/home/og/home/lqx/code/FaceAdaptation')

import cv2
from tqdm import tqdm
import numpy as np

import argparse
from imutils import face_utils
from retinaface.pre_trained_models import get_model
import torch
from datasets.celebdf import get_celeb_mode_list

# 确认裁剪的文件夹
# 使用retinaface 对图片中人脸进行预测
#
def retinaface_crop(model, org_path, save_path, period=1, num_frames=10, args=None):

    if args.dataset in ['DFDCP', 'DFDCP_test']:
        cap_org = cv2.VideoCapture(os.path.join(save_path, org_path))
        mask_cap = None
    else:
        cap_org = cv2.VideoCapture(org_path)
        mask_cap = None 
    
    croppedfaces = []
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=np.int)
    for cnt_frame in range(frame_count_org):
        ret_org, frame_org = cap_org.read()
            
        height, width = frame_org.shape[:-1]
        if not ret_org:
            tqdm.write('Frame read {} Error! : {}'.format(cnt_frame, os.path.basename(org_path)))
            continue

        if cnt_frame not in frame_idxs:
            continue
        else:
            frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)

            faces = model.predict_jsons(frame)
            try:
                if len(faces) == 0:
                    print(faces)
                    tqdm.write('No faces in {}:{}'.format(cnt_frame, os.path.basename(org_path)))
                    continue
                face_s_max = -1
                landmarks = []
                size_list = []
                boxes = []
                for face_idx in range(len(faces)):
                    x0, y0, x1, y1 = faces[face_idx]['bbox']
                    landmark = np.array([[x0, y0], [x1, y1]] + faces[face_idx]['landmarks'])
                    face_size = (x1 - x0) * (y1 - y0)
                    w, h = (x1-x0), (y1-y0)
                    boxes.append([x0,y0,w,h])
                    size_list.append(face_size)
                    landmarks.append(landmark)
            except Exception as e:
                print(f'error in {cnt_frame}:{org_path}')
                print(e)
                continue

        max_idx = np.argmax(size_list)
        # save max faces in the image
        if args.dataset in ['FFIW', 'FFIW_test']:
            boxes = np.array(boxes)
        else:
            boxes = np.array(boxes[max_idx])
        
        if args.dataset in ['DFDCP', 'DFDCP_test']:
            save_path_ = os.path.join(save_path, org_path.replace('.mp4', '_frames/'))
        else:
            save_path_ = org_path.replace('.mp4', '_frames/')
        land_path_ = save_path_.replace('frames', 'retinaface')
        os.makedirs(save_path_, exist_ok=True)
        os.makedirs(land_path_, exist_ok=True)

        image_path = save_path_ + str(cnt_frame).zfill(3) + '.jpg'
        land_path = land_path_ + str(cnt_frame).zfill(3)

        np.save(land_path, boxes)

        if not os.path.isfile(image_path):
            cv2.imwrite(image_path, frame_org)

    cap_org.release()
    return


def filter_vids_from_mode(vids, mode_list=None):
    if mode_list is None:
        return vids
    else:
        results = []
        for v in vids:
            if v.split('/')[-1][:-4] in mode_list:
                results.append(v)
        return results


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset',
                        choices=['DeepFakeDetection_original', 'DeepFakeDetection', 'FaceShifter', 'Face2Face',
                                 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Original', 'Celeb-real', 'Celeb-synthesis',
                                 'YouTube-real', 'DFDCP_test', 'DFDC', 'DFDCP', 'FFIW', 'FFIW_test'])
    parser.add_argument('-c', dest='comp', choices=['raw', 'c23', 'c40'], default='raw')
    parser.add_argument('-n', dest='num_frames', type=int, default=8)
    parser.add_argument('-s', dest='set', default='train')
    args = parser.parse_args()
    if args.dataset == 'Original':
        dataset_path = '/home/og/home/lqx/datasets/FF++/original_sequences/youtube/{}/'.format(args.comp)
    elif args.dataset in ['DFDCP', 'DFDCP_test', 'DFDCVal']:
        dataset_path = '/home/og/home/lqx/mnt_point/datasets/DFDCP_password_123456/DFDCP/'
    elif args.dataset in ['FFIW', 'FFIW_test']:
        dataset_path = '/home/og/home/lqx/mnt_point/datasets/FFIW/FFIW10K-v1-release'
    else:
        raise NotImplementedError

    device = torch.device('cuda')

    model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    model.eval()
    # 读取json 文件
    
    if args.dataset == 'DFDCP':
        all_movies = json.load(open(os.path.join(dataset_path, 'dataset.json')))
        all_movies = list(all_movies.keys())

    elif args.dataset == 'DFDCP_test':
        all_movies_data = json.load(open(os.path.join(dataset_path, 'dataset.json')))
        all_movies = []
        for k,v in all_movies_data.items():
            if v['set'] == 'test':
                all_movies.append(k)
        
    elif args.dataset == 'FFIW':
        all_movies = glob(os.path.join(dataset_path, 'source/*/*.mp4'))
        all_movies += glob(os.path.join(dataset_path, 'target/*/*.mp4'))
    elif args.dataset == 'FFIW_test':
        all_movies = glob(os.path.join(dataset_path, 'source/val/*.mp4'))
        all_movies += glob(os.path.join(dataset_path, 'target/val/*.mp4'))
    else:
        all_movies = []

    print("{} : videos are exist in {}".format(len(all_movies), args.dataset))

    n_sample = len(all_movies)

    for i in tqdm(range(n_sample)):
            # folder_path = all_movies[i].replace('.mp4', '_frames/')
        if args.dataset == 'FFIW':
            vid_path = all_movies[i]
        elif args.dataset in ['DFDCP', 'DFDCP_test']:
            vid_path = os.path.join(dataset_path, all_movies[i])
        else:
            vid_path = all_movies[i]

        retinaface_crop(model, all_movies[i], save_path=dataset_path, num_frames=args.num_frames, args=args)
