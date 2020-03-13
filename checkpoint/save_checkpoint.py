from os.path import join
import os


def save_check_point(model_name, epoch=None, correct=None, loss=None, tpr=None, tnr=None):
    if model_name is None:
        raise Exception
    dir_path = join('/root/data/wzy/checkpoint', model_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    with open(dir_path + '_log.txt', 'a+') as file:
        file.write("Epoch:" + str(epoch) + '\t')
        file.write("Correct:" + str(correct) + '\t')
        file.write("Loss:" + str(loss) + '\t')
        file.write("TPR:" + str(tpr) + '\t')
        file.write("TNR:" + str(tnr) + '\n')
    file.close()
