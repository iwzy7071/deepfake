import json
from tqdm import tqdm
from torchvggish import vggish, vggish_input
import numpy as np
import torch
import os
from os.path import join
import multiprocessing

train_dataset = json.load(open('/root/data/wzy/datalocker/audio_train.json'))

error = []


def multitask_process(train_dataset):
    for path, label in tqdm(train_dataset):
        try:
            save_path = path.replace('DFDC_audios', 'DFDC_audios_tenor')
            video_name = save_path.split('/')[-1]
            save_dir = save_path.split(video_name)[0]
            os.makedirs(save_dir, exist_ok=True)
            embedding_model = vggish()
            embedding_model.eval()
            with torch.no_grad():
                example = vggish_input.wavfile_to_examples(path)
                embeddings = embedding_model.forward(example).numpy()
                embeddings = np.array(embeddings)
                np.savetxt(join(save_dir, '{}'.format(video_name)), embeddings)
        except:
            error.append(path)


min_sample = round(len(train_dataset) / 4)
processes = []
for index in range(4):
    end = min((index + 1) * min_sample, len(train_dataset))
    start = index * min_sample
    process = multiprocessing.Process(target=multitask_process, args=(train_dataset[start:end],))
    processes.append(process)
    process.start()

[process.join() for process in processes]
import json

json.dump(error, open('error_file.json', 'w'))
