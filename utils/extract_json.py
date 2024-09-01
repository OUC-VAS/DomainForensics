import os
import json

root = '/home/og/home/lqx/mnt_point/datasets/FF++'

fake_root = '/home/og/home/lqx/mnt_point/datasets/FF++/manipulated_sequences/Deepfakes/c23/videos/'
real_root = '/home/og/home/lqx/mnt_point/datasets/FF++/original_sequences/youtube/c23/videos/'

train_json = os.path.join(root, 'splits', 'train.json')
test_json = os.path.join(root, 'splits', 'test.json')

train_data = json.load(open(train_json))
test_data = json.load(open(test_json))

# print(train_data)

# all_item = []
# for i in range(len(train_data)):
#     idx1, idx2 = train_data[i]
#     all_item.append(os.path.join(fake_root, '_'.join([idx1, idx2])+'.mp4 1 300 1'))
#     all_item.append(os.path.join(fake_root, '_'.join([idx2, idx1])+'.mp4 1 300 1'))
#     all_item.append(os.path.join(real_root, str(idx1)+'.mp4 1 300 0'))
#     all_item.append(os.path.join(real_root, str(idx2)+'.mp4 1 300 0'))

# with open('./train.txt', 'w') as f:
#     for i in range(len(all_item)):
#         f.write(all_item[i] + '\n')
        

all_item = []
for i in range(len(test_data)):
    idx1, idx2 = test_data[i]
    all_item.append(os.path.join(fake_root, '_'.join([idx1, idx2])+'.mp4 1 300 1'))
    all_item.append(os.path.join(fake_root, '_'.join([idx2, idx1])+'.mp4 1 300 1'))
    all_item.append(os.path.join(real_root, str(idx1)+'.mp4 1 300 0'))
    all_item.append(os.path.join(real_root, str(idx2)+'.mp4 1 300 0'))

with open('./test.txt', 'w') as f:
    for i in range(len(all_item)):
        f.write(all_item[i] + '\n')


    
