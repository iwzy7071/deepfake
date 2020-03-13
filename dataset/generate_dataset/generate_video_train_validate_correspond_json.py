import json
import os
from os.path import join
from tqdm import tqdm

root_path = '/root/data/new_tracking_image_large_factor'

train_result = []
validate_result = []
test_part_names = [9, 18, 28, 36, 48]
fake2real = json.load(open('/root/data/wzy/datalocker/fake2real_vedio.json'))
name2path = json.load(open('/root/data/wzy/datalocker/video_name_to_path.json'))
name2label = json.load(open('/root/data/wzy/datalocker/video_name2label.json'))


def get_correct_frame_path(path):
    root_path = '/'.join(path.split('/')[7:])
    path = join('/root/disk3/DFDC_images', root_path)
    if not os.path.exists(path):
        path = join('/root/disk4/deepfake/dataset/DFDC_images', root_path)
    if not os.path.exists(path):
        return None
    return path


for fake_name, real_name in fake2real.items():
    if fake_name not in name2path or real_name not in name2path:
        continue
    fake_path, real_path = name2path[fake_name], name2path[real_name]
    fake_path, real_path = fake_path.strip('.mp4'), real_path.strip('.mp4')
    fake_path = get_correct_frame_path(fake_path)
    real_path = get_correct_frame_path(real_path)
    if fake_path is None or real_path is None:
        continue

    part_name = int(fake_path.split('/')[-2].split('_')[-1])
    istest = part_name in test_part_names
    if len(os.listdir(fake_path)) < 8 or len(os.listdir(real_path)) < 8:
        continue
    if istest:
        validate_result.append([fake_path, 1])
        validate_result.append([real_path, 0])
    else:
        train_result.append([fake_path, 1])
        train_result.append([real_path, 0])

json.dump(train_result, open('/root/data/wzy/datalocker/frame_linespace_8_train.json', 'w'))
json.dump(validate_result, open('/root/data/wzy/datalocker/frame_linespace_8_validate.json', 'w'))
