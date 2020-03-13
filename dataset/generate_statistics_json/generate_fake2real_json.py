import json
from os.path import join

output = {}
for index in range(0, 50):
    json_path = join('/root/disk4/deepfake/dataset/DFDC_videos/DFDC_videos', 'dfdc_train_part_{}'.format(index),
                     'metadata.json')
    for key, value in json.load(open(json_path)).items():
        if value['label'] != 'FAKE':
            continue
        output[key] = value['original']

json.dump(output, open('/root/data/wzy/datalocker/fake2real_vedio.json', 'w'))
