import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import json
import random
import os
from os.path import join


class PatchAllface(Dataset):
    def __init__(self, txtpath):
        self.data = json.load(open(txtpath))
        random.shuffle(self.data)

    def __getitem__(self, idx):
        video_paths, mask_paths = self.data[idx]
        images = [Image.open(path).convert('RGB') for path in video_paths]
        images = [transform(image) for image in images]
        if '<mask>' in mask_paths[0]:
            label = 0
            masks = [torch.ones([3, 32, 32]) for _ in range(len(mask_paths))]
        else:
            label = 1
            masks = [Image.open(path).convert('RGB') for path in mask_paths]
            masks = [transform_mask(mask) for mask in masks]

        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        return images, masks, label

    def __len__(self):
        return len(self.data)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_mask = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def get_dataloader(BATCH_SIZE):
    data_image = PatchAllface('/root/data/wzy/datalocker/multitask_train_file.json')
    train_dataloader = DataLoader(dataset=data_image, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)
    data_image = PatchAllface('/root/data/wzy/datalocker/multitask_test_file.json')
    test_dataloader = DataLoader(dataset=data_image, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    get_dataloader(256)
