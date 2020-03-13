import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import os
from sklearn.metrics import recall_score, accuracy_score
import torch.backends.cudnn as cudnn
import torch.optim as optim
from checkpoint.save_checkpoint import save_check_point
from dataloader.dataloader_face_corredpond_256 import get_dataloader
from os.path import join
from model.mictnet import get_mictresnet

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
cudnn.benchmark = True

best_wP = 0
start_epoch = 3

print('==> Preparing data..')

BATCH_SIZE = 144
EPOCH = 120
INTERVAL = 400
model_name = 'mict'
save_path = join("/root/data/wzy/checkpoint", model_name)
train_dataloader, test_dataloader = get_dataloader(BATCH_SIZE)

print('==> Building model..')

net = get_mictresnet(backbone='resnet34', version='v2')
net.fc = nn.Linear(net.fc.in_features, 2, bias=True)
net.load_state_dict(torch.load('/root/data/wzy/checkpoint/mict/2_0.7839120120767737.pth'))

# unfreeze_list = ['fc']
# for name, subnet in net.named_children():
#     if name not in unfreeze_list:
#         subnet.requires_grad = False
print("Full Training")
net = nn.DataParallel(net).cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1, last_epoch=-1)

loss_fn = torch.nn.CrossEntropyLoss()


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0
    true_label = []
    pred_label = []
    pred_score = []
    batch_idx = 0
    for video_images, video_label in tqdm(train_dataloader):
        batch_idx += 1
        video_images = video_images.view(video_images.size()[0] * 2, 8, 3, 256, 256)
        video_label = video_label.view(video_label.size()[0] * 2)
        video_images = video_images.permute(0, 2, 1, 3, 4)
        video_images, video_label = Variable(video_images.cuda()), Variable(video_label.cuda())
        optimizer.zero_grad()

        y_pred = net(video_images)
        loss = loss_fn(y_pred, video_label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y_pred.max(1)
        total += video_label.size(0)
        correct += predicted.eq(video_label).sum().item()
        pred_label += list(predicted.cpu().numpy())
        true_label += list(video_label.cpu().numpy())
        pred_score += list(y_pred[:, 1].cpu().detach().numpy())

        if batch_idx % INTERVAL == 0:
            print("Train-Batch:", batch_idx,
                  "correct:", correct / (batch_idx * BATCH_SIZE) / 2,
                  "Loss:", train_loss / (batch_idx * BATCH_SIZE),
                  "TPR:", recall_score(true_label, pred_label, pos_label=1),
                  "TNR:", recall_score(true_label, pred_label, pos_label=0),
                  "Acc:", accuracy_score(true_label, pred_label))


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    true_label = []
    pred_label = []
    pred_score = []
    batch_idx = 0
    with torch.no_grad():
        for video_images, video_label in tqdm(test_dataloader):
            batch_idx += 1
            video_images = video_images.view(video_images.size()[0] * 2, 8, 3, 256, 256)
            video_label = video_label.view(video_label.size()[0] * 2)
            video_images = video_images.permute(0, 2, 1, 3, 4)
            video_images, video_label = Variable(video_images.cuda()), Variable(video_label.cuda())
            y_pred = net(video_images)
            loss = loss_fn(y_pred, video_label)
            test_loss += loss.item()

            _, predicted = y_pred.max(1)
            total += video_label.size(0)
            correct += predicted.eq(video_label).sum().item()
            pred_label += list(predicted.cpu().numpy())
            true_label += list(video_label.cpu().numpy())
            pred_score += list(y_pred[:, 1].cpu().detach().numpy())

    loss = test_loss / (batch_idx * BATCH_SIZE)
    tpr = recall_score(true_label, pred_label, pos_label=1)
    tnr = recall_score(true_label, pred_label, pos_label=0)
    acc = accuracy_score(true_label, pred_label)

    save_check_point(model_name, epoch, acc, loss, tpr, tnr)
    torch.save(net.module.state_dict(), join(save_path, "{}_{}.pth".format(epoch, acc)))


for epoch in range(start_epoch, start_epoch + EPOCH):
    train(epoch)
    test(epoch)
    scheduler.step()
