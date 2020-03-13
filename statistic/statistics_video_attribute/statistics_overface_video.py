import json
import glob
from tqdm import tqdm

name2label = json.load(open('/root/data/wzy/datalocker/video_name2label.json'))
filenames = glob.glob('/root/disk2/tracking_landmarks/DFDC_face_tracking/*/*.json')
face_number_video = {}

for filename in tqdm(filenames):
    json_file = json.load(open(filename))
    video_name = filename.split('/')[-1]
    video_name = video_name.replace('json', 'mp4')
    if video_name not in name2label:
        continue
    label = name2label[video_name]
    face_number_video.setdefault(len(json_file), {}).setdefault(label, []).append(video_name)

json.dump(face_number_video, open('/root/data/wzy/datalocker/video_label_number_statistics.json', 'w'))
