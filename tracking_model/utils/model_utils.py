# --------------------------------------------------------
# Model Utils
# Licensed under The MIT License
# Written by limengyao(mengyao.lmy@alibaba-inc.com)
# --------------------------------------------------------
#!/usr/bin/python

import os
import torch
import logging


def load_check_point(net, filename, device):
    if os.path.isfile(filename):
        logging.info('==> loading checkpoint from %s', filename)

        model_dict = net.state_dict()
        checkpoint_state_dict = torch.load(filename, map_location=device)
        checkpoint_state_dict = checkpoint_state_dict['state_dict']
        checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_dict}
        model_dict.update(checkpoint_state_dict)
        net.load_state_dict(model_dict)
        return net
    else:
        logging.info('==> no checkpoint found at %s', filename)
        return None


def load_model(net, filename, device):
    if os.path.isfile(filename):
        logging.info('==> loading model from %s', filename)
        model = torch.load(filename, map_location=device)
        net.load_state_dict(model)
        return net
    else:
        logging.info('==> no model found at %s', filename)
        return None
