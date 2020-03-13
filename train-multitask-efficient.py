import torch
from tqdm import tqdm
from torch.autograd import Variable
import os
from sklearn.metrics import recall_score, accuracy_score
import torch.backends.cudnn as cudnn
import torch.optim as optim
from dataloader.dataloader_multitask_256 import get_dataloader
from os.path import join
from model.efficient_multitask.efficient_multitask import get_efficient_attention
import argparse
import lr_scheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True if device == 'cuda' else False

BATCH_SIZE = 32
EPOCH = 20
INTERVAL = 200
model_name = 'efficient_with_multitask_new'
save_path = join("/root/data/wzy/checkpoint", model_name)
os.makedirs(save_path, exist_ok=True)
train_dataloader, test_dataloader = get_dataloader(BATCH_SIZE)

print('==> Building model..')
net = get_efficient_attention()

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

clf_loss = torch.nn.CrossEntropyLoss()
att_loss = torch.nn.SmoothL1Loss()

optimizer = optim.Adam(net.parameters(), lr=args.lr)
lr_scheduler = lr_scheduler.build_lr_scheduler(args, optimizer)
train_batch_idx = 0


def train(epoch):
    print("Current epoch {}".format(epoch))
    global train_batch_idx
    net.train()
    train_classify_loss, train_attention_loss = 0.0, 0.0

    true_label = []
    pred_label = []
    pred_score = []
    batch_idx = 0

    for face, mask, label in tqdm(train_dataloader):
        batch_idx += 1
        train_batch_idx += 1
        mask = mask.view(mask.size(0) * mask.size(1), 3, 32, 32)
        face, mask, label = Variable(face.cuda()), Variable(mask.cuda()), Variable(label.cuda())
        optimizer.zero_grad()
        y_pred, attention = net(face)

        mask = mask[:, 0, :, :]
        mask[mask > 0] = 1
        mask[mask < 0] = -1
        mask = mask.unsqueeze(dim=1)
        classify_loss = clf_loss(y_pred, label)
        attention_loss = att_loss(attention, mask)
        loss = classify_loss + attention_loss
        loss.backward()
        optimizer.step()

        train_classify_loss += classify_loss.item()
        train_attention_loss += attention_loss.item()

        _, predicted = y_pred.max(1)
        pred_label += list(predicted.cpu().numpy())
        true_label += list(label.cpu().numpy())
        pred_score += list(y_pred[:, 1].cpu().detach().numpy())
        current_lr = lr_scheduler.step_update(train_batch_idx)

        if batch_idx % INTERVAL == 0:
            print("Train-Batch:", batch_idx,
                  "lr", current_lr,
                  "CLF_Loss:", train_classify_loss / (batch_idx * BATCH_SIZE),
                  "ATT_Loss:", train_attention_loss / (batch_idx * BATCH_SIZE),
                  "TPR:", recall_score(true_label, pred_label, pos_label=1),
                  "TNR:", recall_score(true_label, pred_label, pos_label=0),
                  "Acc:", accuracy_score(true_label, pred_label))


def test(epoch):
    net.eval()
    train_classify_loss, train_attention_loss = 0.0, 0.0

    true_label = []
    pred_label = []
    pred_score = []
    batch_idx = 0

    with torch.no_grad():
        for face, mask, label in tqdm(test_dataloader):
            mask = mask.view(mask.size(0) * mask.size(1), 3, 32, 32)
            batch_idx += 1
            face, mask, label = Variable(face.cuda()), Variable(mask.cuda()), Variable(label.cuda())
            face, label = Variable(face.cuda()), Variable(label.cuda())
            y_pred, attention = net(face)
            mask = mask[:, 0, :, :]
            mask[mask > 0] = 1
            mask[mask < 0] = -1
            mask = mask.unsqueeze(dim=1)
            classify_loss = clf_loss(y_pred, label)
            attention_loss = att_loss(attention, mask)

            train_classify_loss += classify_loss.item()
            train_attention_loss += attention_loss.item()

            _, predicted = y_pred.max(1)
            pred_label += list(predicted.cpu().numpy())
            true_label += list(label.cpu().numpy())
            pred_score += list(y_pred[:, 1].cpu().detach().numpy())

    print("TEST-Batch:", batch_idx,
          "CLF_Loss:", train_classify_loss / (batch_idx * BATCH_SIZE),
          "ATT_Loss:", train_attention_loss / (batch_idx * BATCH_SIZE),
          "TPR:", recall_score(true_label, pred_label, pos_label=1),
          "TNR:", recall_score(true_label, pred_label, pos_label=0),
          "Acc:", accuracy_score(true_label, pred_label))

    torch.save(net.module.state_dict(),
               join(save_path, "{}_{}.pth".format(epoch, accuracy_score(true_label, pred_label))))


for epoch in range(0, 20):
    train(epoch)
    test(epoch)
