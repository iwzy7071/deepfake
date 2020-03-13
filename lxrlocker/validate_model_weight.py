import torch

wzy_model = torch.load('/root/data/wzy/checkpoint/efficient-all-frame-new/5.pth')
lxr_model = torch.load('/root/data/wzy/lxrlocker/efficient-Acc0.877-All-Frame-45part-best-acc.pth')
for key, value in wzy_model.items():
    key = 'backbone.' + key
    lxr_value = lxr_model[key]
    diff = (lxr_value.cpu() - value.cpu()).sum()
    print(diff)