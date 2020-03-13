fake_result = []
real_result = []

for line in open('/root/data/DFDC_audios/audio_meta_all.txt').readlines():
    audio_label = line.strip('\n').split(',')[-1]
    audio_path = line.strip('\n').split(',')[3]
    if audio_label == 'FAKE':
        fake_result.append(audio_path)
    elif audio_label == 'REAL':
        real_result.append(audio_path)

import random

real_result = random.sample(real_result, 25000)
random.shuffle(real_result)
random.shuffle(fake_result)

test_real_index = round(len(real_result) / 10)
test_fake_index = round(len(real_result) / 10)

train_result = []
test_result = []

from os.path import join

for audio in real_result[:test_real_index]:
    path = join('/root/data', audio)
    test_result.append([path, 0])

for audio in fake_result[:test_fake_index]:
    path = join('/root/data', audio)
    test_result.append([path, 1])

for audio in real_result[test_real_index:]:
    path = join('/root/data', audio)
    train_result.append([path, 0])

for audio in fake_result[test_fake_index:]:
    path = join('/root/data', audio)
    train_result.append([path, 1])

import json

json.dump(train_result, open(join('/root/data/wzy/datalocker', 'audio_train.json'),'w'))
json.dump(test_result, open(join('/root/data/wzy/datalocker', 'audio_test.json'),'w'))