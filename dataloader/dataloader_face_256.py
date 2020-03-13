import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import json
import os
from os.path import join
import torch
import numpy as np


class PatchAllface(Dataset):
    def __init__(self, txtpath, transform):
        data_json = json.load(open(txtpath))
        self.videopaths = list(data_json.keys())
        self.labels = list(data_json.values())
        self.transform = transform

    def __getitem__(self, idx):
        videopath, label = self.videopaths[idx], self.labels[idx]
        image_names = sorted(os.listdir(videopath), key=lambda x: int(x.split('_')[0]))
        image_nums = np.linspace(0, len(image_names) - 1, 10, dtype=int, endpoint=True).tolist()
        image_names = [image_names[image_num] for image_num in image_nums]

        image_paths = [join(videopath, image_name) for image_name in image_names]
        images = [Image.open(imagepath).convert('RGB') for imagepath in image_paths]
        choice = random.randint(0, 6)
        if choice == 0 or choice == 1:
            new_images = []
            for image in images:
                w, h = image.size[0], image.size[1]
                transform = transforms.Resize((w // 2, h // 2))
                new_images.append(transform(image))
            images = new_images

        images = [self.transform(image) for image in images]
        image = torch.stack(images, dim=0)
        return image, label

    def __len__(self):
        return len(self.labels)


transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def get_dataloader(BATCH_SIZE):
    datatrain = "/root/data/wzy/datalocker/large_factor_train.json"
    dataval = "/root/data/wzy/datalocker/large_factor_validate.json"
    data_image = PatchAllface(txtpath=datatrain, transform=transform_train)
    data_loader_image = DataLoader(dataset=data_image, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)
    test_data = PatchAllface(txtpath=dataval, transform=transform_test)
    test_data_loader = DataLoader(dataset=test_data, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)

    return data_loader_image, test_data_loader


if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloader(32)
    for image, label in train_dataloader:
        print(image.size())
        pass
