# --------------------------------------------------------
# Tracker Utils
# Licensed under The MIT License
# Written by limengyao(mengyao.lmy@alibaba-inc.com)
# --------------------------------------------------------
# !/usr/bin/python
import os
import torch
import numpy as np
import logging


def cxy_wh_2_rect(pos, sz):
    return np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, sz[0], sz[1]])  # 0-index


def cxy_wh_2_corner(pos, sz):
    return np.array([pos[0] - sz[0] / 2, pos[1] - sz[1] / 2, pos[0] + sz[0] / 2, pos[1] + sz[1] / 2])


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


class Tracker:
    def __init__(self, track_method, net_name=None, model=None):
        """

        :param track_method: 0: dcf method; 1: dl method
        :param net_name:
        :return:
        """
        self.supported_trackers = {0: 'SiamRPN', 1: 'ECO', 2: 'KCF', }
        self.dl_trackers = ['SiamRPN']
        self.cv_trackers = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.tracker_type = self.supported_trackers[track_method]
        self.state = None
        self.device = None

        if self.tracker_type in self.dl_trackers:
            if net_name is None or model is None:
                raise FileNotFoundError('invalid network!')
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
            from tracking_model.trackers.siamrpn.net.dasiamrpn import SiamRPNotb
            from tracking_model.trackers.siamrpn.net.dasiamrpn_batch_distillation_net import SiamRPNbatchOTBMobile, \
                SiamRPNbatchOTB

            net = locals()[net_name]()
            param_size = np.sum(np.prod(v.size()) for v in net.parameters()) / 1e6
            logging.info('Total parameter size %f M', param_size)

            if not model.find('.model') == -1:
                net = load_model(net, model, self.device)
            else:
                net = load_check_point(net, model, self.device)
            net.eval().to(self.device)
            self.tracker = net

        elif self.tracker_type in self.cv_trackers:
            from tracking_model.trackers.cv_tracker import create_cv_tracker
            self.tracker = create_cv_tracker(self.tracker_type)

    def init_tracker(self, image, target_pos, target_sz):
        """

        :param target_pos: target center, nparray
        :param target_sz: target width and height, nparray
        :return: init result, True or False
        """
        if self.tracker_type == 'SiamRPN':
            from tracking_model.trackers.siamrpn.eval.run_SiamRPN import SiamRPN_init
            self.state = SiamRPN_init(image, target_pos, target_sz, self.tracker, self.device)
            return True
        elif self.tracker_type in self.cv_trackers:
            from tracking_model.trackers.cv_tracker import init_cv_tracker
            bbox = cxy_wh_2_corner(target_pos, target_sz)
            return init_cv_tracker(self.tracker, image, bbox)

    def update_tracker(self, image):
        """

        :param image: detection image
        :return: location
        """

        if self.tracker_type == 'SiamRPN':
            from tracking_model.trackers.siamrpn.eval.run_SiamRPN import SiamRPN_track
            state = SiamRPN_track(self.state, image, self.device)  # track
            location = cxy_wh_2_rect(state['target_pos'] + 1, state['target_sz'])
            return location, state['score']
        if self.tracker_type in self.cv_trackers:
            from tracking_model.trackers.cv_tracker import update_cv_tracker
            location = update_cv_tracker(self.tracker, image)
            return location
