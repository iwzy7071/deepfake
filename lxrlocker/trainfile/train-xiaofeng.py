from dataset import DFDC_image_dataset
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
from model.efficientnet.model2 import EfficientNet
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
import os
from collections.abc import Iterable
import datetime
import time

batchsize = 32
frame_num = 8
log_interval = 100
model_name = 'mean'
use_whole_frame = False
resume = ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(8848)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



class DFDC_Model(nn.Module):

    def __init__(self, embedding_size=1792, hidden_size=256, num_classes=2):
            super(DFDC_Model, self).__init__()
            self.embedding_size = embedding_size
            self.cnn = EfficientNet.from_pretrained("efficientnet-b4", num_classes=2)
            self.last_linear = nn.Linear(embedding_size, num_classes)
            self.dropout = nn.Dropout(0.2)

    def forward(self, x):
            bs, fs, C, H, W = x.size()
            c_in = x.view(bs * fs, C, H, W)
            c_out, _ = self.cnn(c_in)
            r_out = torch.mean(c_out.view(bs, fs, -1), dim=1)
            r_out = self.dropout(r_out)
            r_out2 = self.last_linear(r_out)
            return r_out2


model = DFDC_Model()
if resume != '':
    gpu_id=0
    model_dict = model.state_dict()
    kwargs = {'map_location': lambda storage, loc: storage.cuda(gpu_id)}
    state_dict = torch.load(resume, **kwargs)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

model.to(device)
model = nn.DataParallel(model)

val_split = [9, 18, 28, 36, 48]
train_split = [x for x in range(50)]

train_dataset = DFDC_image_dataset(meta_path='/root/disk4/deepfake/dataset/DFDC_videos/DFDC_videos',
                                   face_info_path='/root/DFDC_face_tracking',
                                   sample_num=frame_num, split=train_split, transform=train_transform,
                                   whole_frame=use_whole_frame)
train_size = len(train_dataset)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batchsize, num_workers=48, shuffle=True)

val_dataset = DFDC_image_dataset(meta_path='/root/disk4/deepfake/dataset/DFDC_videos/DFDC_videos',
                                 face_info_path='/root/DFDC_face_tracking',
                                 sample_num=frame_num, split=val_split, transform=val_transform,
                                 whole_frame=use_whole_frame)
val_size = len(val_dataset)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=batchsize, num_workers=48, shuffle=False)

optimizer = optim.Adam(model.parameters(),
                       lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
loss_fun = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(0,20):
    correct_tot = 0
    epoch_loss = 0
    model.train()
    for batch_idx, (face, label) in enumerate(train_loader):
        optimizer.zero_grad()
        face, label = face.to(device), label.to(device)
        output = model(face)
        pred = output.max(1, keepdim=True)[1]
        correct_tot += pred.eq(label.view_as(pred)).sum().item()
        loss = loss_fun(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                epoch, batch_idx *
                       batchsize, train_size,
                       100. * batch_idx / len(train_loader), loss.item(), correct_tot / ((batch_idx + 1) *
                                                                                         batchsize)))
    scheduler.step()
    model.eval()
    # score = 0
    correct = 0
    with torch.no_grad():
        for val_face, val_label in tqdm(val_loader):
            val_face, val_label = val_face.to(device), val_label.to(device)
            out = model(val_face)
            # val_loss = loss_fun(out, val_label)
            pred = out.max(1, keepdim=True)[1]
            correct += pred.eq(val_label.view_as(pred)).sum().item()
            # score += val_loss.item() * val_label.size(0)

        print('*' * 50)
        if correct / val_size > best_acc:
            best_acc = correct / val_size
            print('Imporved. Best acc: {:.4f}'.format(best_acc))
            torch.save(model.state_dict(),
                       os.path.join('/root/data/lxr/kaggle/lxrmodel', "model_epoch_{}_acc_{:.4f}_best.pth".format(epoch, best_acc)))
        else:
            print('Not improved. acc: {:.4f}'.format(correct / val_size))
            torch.save(model.state_dict(),
                       os.path.join('/root/data/lxr/kaggle/lxrmodel', "model_epoch_{}_acc_{:.4f}.pth".format(epoch, correct / val_size)))
        print('*' * 50)
