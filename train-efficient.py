import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import os
from sklearn.metrics import recall_score, accuracy_score
import torch.backends.cudnn as cudnn
import torch.optim as optim
from checkpoint.save_checkpoint import save_check_point
from dataloader.dataloader_frame_299 import get_dataloader
from os.path import join
from model.efficientnet.model import EfficientNet
import lr_scheduler
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
cudnn.benchmark = True
best_wP = 0
start_epoch = 0

print('==> Preparing data..')

BATCH_SIZE = 32
EPOCH = 30
INTERVAL = 200
model_name = 'efficient-all-frame-new'
save_path = join("/root/data/wzy/checkpoint", model_name)
train_dataloader, test_dataloader = get_dataloader(BATCH_SIZE)

print('==> Building model..')

net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2).cuda()
net = nn.DataParallel(net).cuda()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr-scheduler', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
args = parser.parse_args()

args.warmup_init_lr = 1e-5
args.lr = 2e-4
args.max_lr = args.lr
args.warmup_updates = 5 * len(train_dataloader)
args.max_update = EPOCH * len(train_dataloader)
args.epoch = EPOCH

optimizer = optim.Adam(net.parameters(), lr=args.lr)
lr_scheduler = lr_scheduler.build_lr_scheduler(args, optimizer)
loss_fn = torch.nn.CrossEntropyLoss()

train_batch_idx = 0


def train(epoch):
    global train_batch_idx
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
        train_batch_idx += 1
        batch_idx += 1
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
        current_lr = lr_scheduler.step_update(train_batch_idx)

        if batch_idx % INTERVAL == 0:
            print("Train-Batch:", batch_idx,
                  "lr", current_lr,
                  "correct:", correct / (batch_idx * BATCH_SIZE),
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
    torch.save(net.module.state_dict(), join(save_path, str(epoch) + ".pth"))


for epoch in range(start_epoch, start_epoch + EPOCH):
    train(epoch)
    test(epoch)
