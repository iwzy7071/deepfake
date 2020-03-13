import json
import matplotlib.pyplot as plt
import pandas as pd
from math import log


def get_crop_value(preds):
    wzy_logloss = 0
    lxr_logloss = 0
    for label, wzy_prob, lxr_prob in preds:
        wzy_logloss += label * log(wzy_prob) + log(1 - wzy_prob) * (1 - label)
        lxr_logloss += label * log(lxr_prob) + log(1 - lxr_prob) * (1 - label)
    wzy_logloss = wzy_logloss / (-len(preds))
    lxr_logloss = lxr_logloss / (-len(preds))
    return wzy_logloss, lxr_logloss


global_statistic = json.load(open('/root/data/wzy/statistic/offline_with_global_detect.json'))
global_tracking = json.load(open('/root/data/wzy/statistic/offline_tracking_with_global_detect.json'))
global_tracking_local_detect = json.load(open('/root/data/wzy/statistic/offline_tracking_with_global_area_detect.json'))

# global_statistic_logloss = get_crop_value(global_statistic)
# global_tracking_logloss = get_crop_value(global_tracking)
# global_tracking_local_detect_logloss = get_crop_value(global_tracking_local_detect)

def get_acc(preds):
    wzy_acc = 0
    lxr_acc = 0
    for label, wzy_prob, lxr_prob in preds:
        wzy_prob = 1 if wzy_prob > 0.5 else 0
        lxr_prob = 1 if lxr_prob > 0.5 else 0
        wzy_acc += 1 if wzy_prob == label else 0
        lxr_acc += 1 if lxr_prob == label else 0
    wzy_logloss = wzy_acc / len(preds)
    lxr_logloss = lxr_acc / len(preds)
    return wzy_logloss, lxr_logloss

global_statistic_logloss = get_acc(global_statistic)
global_tracking_logloss = get_acc(global_tracking)
global_tracking_local_detect_logloss = get_acc(global_tracking_local_detect)

print(global_statistic_logloss)
print(global_tracking_logloss)
print(global_tracking_local_detect_logloss)

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(9, 6))
ax0.bar(x=['wzy', 'lxr'], height=global_statistic_logloss, width=0.35, color='r', alpha=0.75)
ax0.set_title('global detect without tracking')
ax1.bar(x=['wzy', 'lxr'], height=global_tracking_logloss, width=0.35, color='g')
ax1.set_title('global detect with tracking')
ax2.bar(x=['wzy', 'lxr'], height=global_tracking_local_detect_logloss, width=0.35, color='b')
ax2.set_title('global detect with tracking and local detect')
fig.subplots_adjust(hspace=0.4)
plt.savefig('Acc_distribution.png')
