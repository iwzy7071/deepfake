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
        self.data_json = json.load(open(txtpath))
        self.transform = transform

    def __getitem__(self, idx):
        video_paths = self.data_json[idx]
        fake_video_path, real_video_path = video_paths

        fake_image_names = os.listdir(fake_video_path)
        fake_image_nums = np.linspace(0, len(fake_image_names) - 1, 8, dtype=int).tolist()
        fake_image_names = [fake_image_names[num] for num in fake_image_nums]
        fake_image_paths = [join(fake_video_path, image_name) for image_name in fake_image_names]
        fake_images = [Image.open(imagepath).convert('RGB') for imagepath in fake_image_paths]

        real_image_names = os.listdir(real_video_path)
        real_image_nums = np.linspace(0, len(real_image_names) - 1, 8, dtype=int).tolist()
        real_image_names = [real_image_names[num] for num in real_image_nums]
        real_image_paths = [join(real_video_path, image_name) for image_name in real_image_names]
        real_images = [Image.open(imagepath).convert('RGB') for imagepath in real_image_paths]
        choice = random.randint(0, 1)
        if choice == 0:
            new_images = []
            for image in real_images:
                w, h = image.size[0], image.size[1]
                transform = transforms.Resize((w // 2, h // 2))
                new_images.append(transform(image))
            real_images = new_images

        choice = random.randint(0, 1)
        if choice == 0:
            new_images = []
            for image in fake_images:
                w, h = image.size[0], image.size[1]
                transform = transforms.Resize((w // 2, h // 2))
                new_images.append(transform(image))
            fake_images = new_images

        fake_images = [self.transform(image) for image in fake_images]
        real_images = [self.transform(image) for image in real_images]

        fake_images = torch.stack(fake_images, dim=0)
        real_images = torch.stack(real_images, dim=0)

        if random.randint(0, 1) == 0:
            images = torch.stack((real_images, fake_images), dim=0)
            label = torch.tensor([0, 1])
        else:
            images = torch.stack((fake_images, real_images), dim=0)
            label = torch.tensor([1, 0])
        return images, label

    def __len__(self):
        return len(self.data_json)


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
    datatrain = "/root/data/wzy/datalocker/corres_large_factor_train.json"
    dataval = "/root/data/wzy/datalocker/corres_large_factor_validate.json"
    data_image = PatchAllface(txtpath=datatrain, transform=transform_train)
    data_loader_image = DataLoader(dataset=data_image, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)
    test_data = PatchAllface(txtpath=dataval, transform=transform_test)
    test_data_loader = DataLoader(dataset=test_data, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)

    return data_loader_image, test_data_loader


if __name__ == '__main__':
    train_dataloader, test_dataloader = get_dataloader(16)
    for video_images, video_label in train_dataloader:
        video_images = video_images.view(16 * 2, 8, 3, 256, 256)
        video_label = video_label.view(16 * 2)
        video_images = video_images.permute(0, 2, 1, 3, 4)
        print(video_images.size())
        exit()
