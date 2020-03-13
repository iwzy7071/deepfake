import shutil
import json
import random

name2_path = json.load(open('/root/data/wzy/datalocker/video_name_to_path.json'))
root_path = '/root/three_kinds_image_effect_on_model'
video_dataset = json.load(open('/root/data/wzy/datalocker/video_label_number_statistics.json'))
final_video_dataset = {}
final_video_dataset['person_5_FAKE'] = video_dataset['5']['FAKE']
final_video_dataset['person_5_REAL'] = video_dataset['5']['REAL']
final_video_dataset['person_4_FAKE'] = random.sample(video_dataset['4']['FAKE'], 30)
final_video_dataset['person_4_REAL'] = video_dataset['4']['REAL']
final_video_dataset['person_3_FAKE'] = random.sample(video_dataset['3']['FAKE'], 30)
final_video_dataset['person_3_REAL'] = random.sample(video_dataset['3']['REAL'], 30)
final_video_dataset['person_2_FAKE'] = random.sample(video_dataset['2']['FAKE'], 30)
final_video_dataset['person_2_REAL'] = random.sample(video_dataset['2']['REAL'], 30)
final_video_dataset['person_1_FAKE'] = random.sample(video_dataset['1']['FAKE'], 30)
final_video_dataset['person_1_REAL'] = random.sample(video_dataset['1']['REAL'], 30)
import os

result = {}
for key, videos in final_video_dataset.items():
    for video_name in videos:
        video_path = name2_path[video_name]
        if os.path.exists(video_path):
            result.setdefault(key, []).append(video_path)

json.dump(result, open('/root/data/wzy/datalocker/three_kinds_video_dataset.json', 'w'))
