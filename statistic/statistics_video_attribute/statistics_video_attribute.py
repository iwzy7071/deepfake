import pandas as pd
import json
from collections import Counter

df = pd.read_csv('/root/data/wzy/statistic/metadata.json')
md5s = df['md5'].value_counts()
md5s = md5s.loc[md5s.values != 1]
md5s = md5s.keys().tolist()

name2path = json.load(open('/root/data/wzy/datalocker/video_name_to_path.json'))
name2label = json.load(open('/root/data/wzy/datalocker/video_name2label.json'))

labels = []
part_name_numbers = []
result = []
for md5 in md5s:
    filenames = df.loc[df.md5 == md5, 'filename'].values.tolist()
    part_names = []

    for filename in filenames:
        result.append(filename)
    #     filepath = name2path[filename]
    #     filelabel = name2label[filename]
    #     filepath = filepath.replace('/root/disk4/deepfake/dataset', '/root/data')
    #     part_name = filepath.split('/')[5]
    #     part_names.append(part_name)
    # part_names = Counter(part_names)
    # print(part_names)
    # part_name_numbers += list(part_names.keys())
# print(Counter(part_name_numbers))
import json

json.dump(result, open('duplicated_videos.json', 'w'))
