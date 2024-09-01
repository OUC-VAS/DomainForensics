from glob import glob
import os

import cv2
from tqdm import tqdm
import numpy as np

import argparse
from imutils import face_utils
from retinaface.pre_trained_models import get_model
import torch


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

def retinaface_crop(model, org_path, save_path, period=1, num_frames=10, args=None):
    cap_org = cv2.VideoCapture(org_path)
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
        boxes = np.array(boxes[max_idx])


        if args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
            save_path_ = save_path.replace(args.dataset, args.dataset+'_frames') + os.path.basename(org_path).replace('.mp4', '/')
        else:
            save_path_ = save_path + 'frames/' + os.path.basename(org_path).replace('.mp4', '/')
        os.makedirs(save_path_, exist_ok=True)
        image_path = save_path_ + str(cnt_frame).zfill(3) + '.jpg'
        land_path = save_path_ + str(cnt_frame).zfill(3)

        if args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
            land_path = land_path.replace('frames', 'retinaface')
        else:
            land_path = land_path.replace('/frames', '/retinaface')
        os.makedirs(os.path.dirname(land_path), exist_ok=True)
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
    
    root = '/path/to/dataset/'
    celeb_root = root + '/Celeb-DF/Celeb-DF-v2/'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset',
                        choices=['DeepFakeDetection_original', 'DeepFakeDetection', 'FaceShifter', 'Face2Face',
                                 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Original', 'Celeb-real', 'Celeb-synthesis',
                                 'YouTube-real', 'DFDC', 'DFDCP'])
    parser.add_argument('-c', dest='comp', choices=['raw', 'c23', 'c40'], default='raw')
    parser.add_argument('-n', dest='num_frames', type=int, default=8)
    parser.add_argument('-s', dest='set', default='train')
    args = parser.parse_args()
    
    
    if args.dataset == 'Original':
        dataset_path = root + '/FF++/original_sequences/youtube/{}/'.format(args.comp)
    elif args.dataset == 'DeepFakeDetection_original':
        dataset_path = root + '/FF++/original_sequences/actors/{}/'.format(args.comp)
    elif args.dataset in ['DeepFakeDetection', 'FaceShifter', 'Face2Face', 'Deepfakes', 'FaceSwap', 'NeuralTextures']:
        dataset_path = root + '/FF++/manipulated_sequences/{}/{}/'.format(args.dataset, args.comp)
    elif args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
        dataset_path = root + '/Celeb-DF/Celeb-DF-v2/{}/'.format(args.dataset)
    elif args.dataset in ['DFDC', 'DFDCVal']:
        dataset_path = 'data/{}/'.format(args.dataset)
    else:
        raise NotImplementedError

    device = torch.device('cuda')

    model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    model.eval()

    if args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
        movies_path = dataset_path
    else:
        movies_path = dataset_path + 'videos/'


    movies_path_list = sorted(glob(movies_path + '*.mp4'))

    

    if args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real'] and args.set == 'train':
        mode_list = get_celeb_mode_list(celeb_root, mode='train')
        movies_path_list = filter_vids_from_mode(movies_path_list, mode_list)
    elif args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real'] and args.set == 'test':
        mode_list = get_celeb_mode_list(celeb_root, mode='test')
        movies_path_list = filter_vids_from_mode(movies_path_list, mode_list)
    else:
        movies_path_list = movies_path_list

    print("{} : videos are exist in {}".format(len(movies_path_list), args.dataset))

    n_sample = len(movies_path_list)

    for i in tqdm(range(n_sample)):
        if args.dataset in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
            folder_path = movies_path_list[i].replace(args.dataset, args.dataset+'_frames').replace('.mp4', '/')
        else:
            folder_path = movies_path_list[i].replace('videos/', 'frames/').replace('.mp4', '/')
        if len(glob(folder_path.replace('frames', 'retinaface') + '*.npy')) < args.num_frames:
            npys = glob(folder_path.replace('frames', 'retinaface') + '*.npy')
            imgs = glob(folder_path + '*.jpg')
            items = npys + imgs
            for f in items:
                os.remove(f)
            retinaface_crop(model, movies_path_list[i], save_path=dataset_path, num_frames=args.num_frames, args=args)

