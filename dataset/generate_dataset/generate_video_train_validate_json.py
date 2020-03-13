import json
import os
from os.path import join
from tqdm import tqdm

json_path = '/root/data/wzy_image_level/datalocker/video_name2label.json'
json_file = json.load(open('/root/data/wzy/datalocker/video_name2label.json'))

root_path1 = '/root/disk4/deepfake/dataset/DFDC_images'
root_path2 = '/root/disk3/DFDC_images'
train_result = {}
validate_result = {}
test_part_names = [9, 18, 28, 36, 48]
for root_path in [root_path1, root_path2]:
    for part_name in tqdm(os.listdir(root_path)):
        istest = int(part_name.split('_')[-1]) in test_part_names
        part_path = join(root_path, part_name)
        for video_name in os.listdir(part_path):
            video_path = join(part_path, video_name)
            if len(os.listdir(video_path)) < 8:
                continue
            video_label = 0 if json_file[video_name + '.mp4'] == 'REAL' else 1
            train_result[video_path] = video_label
            if istest:
                validate_result[video_path] = video_label
            else:
                train_result[video_path] = video_label

json.dump(train_result, open('/root/data/wzy/datalocker/all_frame_train.json', 'w'))
json.dump(validate_result, open('/root/data/wzy/datalocker/all_frame_validate.json', 'w'))
