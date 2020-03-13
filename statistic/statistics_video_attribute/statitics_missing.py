import json
from os.path import join
import os
from tqdm import tqdm

landmasks_root = '/root/disk2/tracking_landmarks/DFDC_face_tracking/'
frames_root = '/root/disk3/DFDC_images'

zero_image_video_list = json.load(open('/root/data/wzy/datalocker/zero_image_video.json'))
supple_video_list = []
count = 0
for video_name in tqdm(zero_image_video_list):
    json_name = video_name + '.json'
    video_json_path = os.popen('find {} -name {}'.format('/root/DFDC_face_tracking', json_name))
    video_json_path = video_json_path.readlines()[0].strip('\n')

    save_path = video_json_path.replace("/root/DFDC_face_tracking",
                                        "/root/disk2/new_tracking_image_huge_factor")
    save_path = save_path[:save_path.index('.')]

    if len(os.listdir(save_path)) > 10:
        supple_video_list.append(save_path)

json.dump(supple_video_list, open('/root/data/wzy/datalocker/supple_video_list.json', 'w'))
