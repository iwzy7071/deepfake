import json
from os.path import join
import shutil

video_paths = json.load(open('/root/data/wzy/datalocker/over_face_number_video.json'))
root_path = '/root/disk4/deepfake/dataset/DFDC_videos/DFDC_videos'
save_path = '/root/download_overface_video'
for video_path in video_paths:
    video_path = '/'.join(video_path.split('/')[-2:])
    video_path = join(root_path, video_path)
    video_path = video_path.replace('json', 'mp4')
    video_name = video_path.split('/')[-1]
    shutil.copy(video_path, join(save_path, video_name))
