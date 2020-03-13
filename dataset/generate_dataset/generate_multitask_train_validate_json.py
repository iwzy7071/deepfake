import os
from os.path import join
import json
from tqdm import tqdm
import numpy as np
import glob

train_result = []
test_result = []
test_part_names = ['dfdc_train_part_9', 'dfdc_train_part_18', 'dfdc_train_part_28', 'dfdc_train_part_36',
                   'dfdc_train_part_48']

name2path = json.load(open('/root/data/wzy/datalocker/video_name_to_path.json'))
fake2real = json.load(open('/root/data/wzy/datalocker/fake2real_vedio.json'))

for fake_name, real_name in fake2real.items():
    if fake_name not in name2path:
        continue
    fake_path = name2path[fake_name]
    part_name = fake_path.split('/')[-2]
    save_fake_path = join('/root/disk2/multitask_mask_2020_02_27/fake', fake_name)
    save_real_path = join('/root/disk2/multitask_mask_2020_02_27/real', real_name)
    save_mask_path = save_fake_path.replace('fake', 'mask')
    if not os.path.exists(save_real_path) or not os.path.exists(save_fake_path) or not os.path.exists(save_mask_path):
        continue

    save_image_names = {}
    image_names = os.listdir(save_fake_path)
    [save_image_names.setdefault(name.split('_')[0], []).append(name) for name in image_names]
    for image_names in save_image_names.values():
        fake_image_paths = [join(save_fake_path, name) for name in image_names]
        real_image_paths = [join(save_real_path, name) for name in image_names]
        fake_mask_image_paths = [join(save_mask_path, name) for name in image_names]
        real_mask_paths = ['<mask>' for name in image_names]
        if part_name in test_part_names:
            test_result.append([fake_image_paths, fake_mask_image_paths])
            test_result.append([real_image_paths, real_mask_paths])
        else:
            train_result.append([fake_image_paths, fake_mask_image_paths])
            train_result.append([real_image_paths, real_mask_paths])


json.dump(train_result, open('/root/data/wzy/datalocker/multitask_train_file.json', 'w'))
json.dump(test_result, open('/root/data/wzy/datalocker/multitask_test_file.json', 'w'))
