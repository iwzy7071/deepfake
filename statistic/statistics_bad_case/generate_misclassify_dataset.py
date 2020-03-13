import os
import json
import shutil
from os.path import join
from tqdm import tqdm

path = "/root/data/wzy/datalocker/large_factor_train.json"
videos = json.load(open(path))
video_name2path = json.load(open('/root/data/wzy/datalocker/video_name_to_path.json'))
for video in tqdm(videos):
    video_name = video.split('/')[-1]
    video_path = video_name2path[video_name + '.mp4']
    video_size = os.path.getsize(video_path)
    images = os.listdir(video)
    for image in images:
        src = join(video, image)
        image_size = os.path.getsize(src) / video_size
        if image_size <= 0.003:
            dst = video.replace("/disk2/", "/disk1/thredhold_example/")
            if not os.path.exists(dst):
                os.makedirs(dst, exist_ok=True)
            dst = join(dst, image)
            shutil.copyfile(src, dst)
