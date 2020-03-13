import json
import matplotlib.pyplot as plt
import pandas as pd
from math import log
import numpy as np
from sklearn.metrics import recall_score, accuracy_score

statistics_value = json.load(open('/root/data/wzy/statistic/value_statistics_distribution.json'))['vpred']
final_score = {}


def get_crop_value(preds):
    logloss = 0
    for score, label in preds:
        logloss += label * log(score) + log(1 - score) * (1 - label)
    logloss = logloss / (-len(preds))
    print(logloss)
    return logloss


# for cropvalue in np.linspace(start=0.95, stop=1, num=11, dtype=float):
cropvalue = 0.9
new_value = []
new_label = []
for current_value in statistics_value:
    score, label = current_value
    # print(score, label)
    if score > 0.5:
        new_value.append(1)
    else:
        new_value.append(0)
    new_label.append(label)

print(len(new_value),len(new_label))
print(accuracy_score(new_label, new_value))
# plt.title('online_pred_distribution')
# plt.savefig('online_pred_distribution.png')
