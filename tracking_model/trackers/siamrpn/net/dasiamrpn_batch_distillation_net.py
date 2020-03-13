# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch


class SiamRPNbatch(nn.Module):
    def __init__(self, size=2, feature_out=512, anchor=5):
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x == 3 else x * size, configs))
        feat_in = configs[-1]
        super(SiamRPNbatch, self).__init__()
        self.featureExtract = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )
        self.config = configs

        self.anchor = anchor
        self.feature_out = feature_out

        self.conv_r1 = nn.Conv2d(feat_in, feature_out * 4 * anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out * 2 * anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4 * anchor, 4 * anchor, 1)

        self.cfg = {}

    def get_config(self):
        return self.config

    def forward(self, x):
        z_f, x_f = self.feature_extract(x)

        features = {}
        features['template'] = z_f
        features['detection'] = x_f

        r1_kernel_raw, cls1_kernel_raw, r2_xf, cls2_xf = self.cf_kernel(z_f, x_f)

        features['r1_kernel_raw'] = r1_kernel_raw
        features['cls1_kernel_raw'] = cls1_kernel_raw
        features['r2_xf'] = r2_xf
        features['cls2_xf'] = cls2_xf
        rout, cout = self.cf_rpn(r1_kernel_raw, cls1_kernel_raw, r2_xf, cls2_xf)

        return rout, cout, features

    def feature_extract(self, x):
        z_f = self.featureExtract(self.z)
        x_f = self.featureExtract(x)
        return z_f, x_f

    def cf_kernel(self, z_f, x_f):
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        r2_xf = self.conv_r2(x_f)
        cls2_xf = self.conv_cls2(x_f)

        return r1_kernel_raw, cls1_kernel_raw, r2_xf, cls2_xf

    def cf_rpn(self, r1_kernel_raw, cls1_kernel_raw, r2_xf, cls2_xf):
        kernel_size = r1_kernel_raw.data.size()[-1]
        r1_kernel = r1_kernel_raw.view(r1_kernel_raw.size()[0], self.anchor * 4, self.feature_out, kernel_size,
                                       kernel_size)
        cls1_kernel = cls1_kernel_raw.view(cls1_kernel_raw.size()[0], self.anchor * 2, self.feature_out, kernel_size,
                                           kernel_size)

        rout_list = []
        cout_list = []
        for rkernel, ckernel, r2_x_feature, c2_x_feature in zip(r1_kernel, cls1_kernel, r2_xf, cls2_xf):
            r2_x_feature = r2_x_feature.unsqueeze(0)
            c2_x_feature = c2_x_feature.unsqueeze(0)
            rout = self.regress_adjust(F.conv2d(r2_x_feature, rkernel))
            cout = F.conv2d(c2_x_feature, ckernel)
            rout_list.append(rout)
            cout_list.append(cout)

        rout = torch.cat(rout_list, 0)
        cout = torch.cat(cout_list, 0)

        return rout, cout

    def temple(self, z):
        self.z = z


class SiamRPNbatchBIG(SiamRPNbatch):
    def __init__(self):
        super(SiamRPNbatchBIG, self).__init__(size=2)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'exemplar_size': 127, 'base_size': 64,
                    'ratios': [0.33, 0.5, 1.0, 2.0, 3.0], 'instance_size': 271, 'adaptive': False}  # 0.355

        # self.cfg = {'lr':0.295, 'window_influence': 0.42, 'penalty_k': 0.055, 'instance_size': 271, 'adaptive': True} # 0.383

        def temple(self, z):
            self.z = z

        def forward(self, x):
            return super.forward(self.z, self.x)


class SiamRPNbatchVOT(SiamRPNbatch):
    def __init__(self):
        super(SiamRPNbatchVOT, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'exemplar_size': 127, 'score_size': 19,
                    'ratios': [0.33, 0.5, 1.0, 2.0, 3.0], 'instance_size': 271, 'adaptive': False}  # 0.355


class SiamRPNbatchOTB(SiamRPNbatch):
    def __init__(self):
        super(SiamRPNbatchOTB, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'exemplar_size': 127,
                    'score_size': 19, 'ratios': [0.33, 0.5, 1.0, 2.0, 3.0], 'scales': [8.0], 'instance_size': 271,
                    'adaptive': False, 'to_tensor': True}  # 0.655


class SiamRPNbatchMobile(SiamRPNbatch):
    def __init__(self, size=2, feature_out=32, anchor=5, configs=[3, 96, 256, 384, 384, 32], teacher_in=256):
        configs = list(map(lambda x: 3 if x == 3 else x * size, configs))
        feat_in = configs[-1]
        super(SiamRPNbatchMobile, self).__init__(size, feature_out, anchor)
        self.featureExtract = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=3, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=3),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        self.config = configs

        self.anchor = anchor
        self.feature_out = feature_out

        self.conv_r1 = nn.Conv2d(feat_in, feature_out * 4 * anchor, 1)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 1)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out * 2 * anchor, 1)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 1)
        self.regress_adjust = nn.Conv2d(4 * anchor, 4 * anchor, 1)

        self.conv_r1_trans = nn.Conv2d(teacher_in * 4 * anchor, feature_out * 4 * anchor, 1)
        self.conv_cls1_trans = nn.Conv2d(teacher_in * 2 * anchor, feature_out * 2 * anchor, 1)
        self.conv_r2_trans = nn.Conv2d(teacher_in, feature_out, 1)
        self.conv_cls2_trans = nn.Conv2d(teacher_in, feature_out, 1)

        self.feature_conv = nn.Conv2d(teacher_in, feat_in, kernel_size=1)

        self.cfg = {}

    def hint(self, teacher_template_feature, teacher_feature):
        return self.feature_conv(teacher_template_feature), self.feature_conv(teacher_feature)

    def hint_rpn_conv(self, r1_kernel_raw, cls1_kernel_raw, r2_xf, cls2_xf):
        return self.conv_r1_trans(r1_kernel_raw), self.conv_cls1_trans(cls1_kernel_raw), \
               self.conv_r2_trans(r2_xf), self.conv_cls2_trans(cls2_xf)

    def hint_kernel(self, r1_kernel_trans, cls1_kernel_trans, r2_xf_trans, cls2_xf_trans):
        return self.cf_rpn(r1_kernel_trans, cls1_kernel_trans, r2_xf_trans, cls2_xf_trans)


class SiamRPNbatchOTBMobile(SiamRPNbatchMobile):
    def __init__(self, teacher_in=256):
        configs = [3, 8, 16, 16, 384, 32]
        super(SiamRPNbatchOTBMobile, self).__init__(size=1, feature_out=16, anchor=5, configs=configs,
                                                    teacher_in=teacher_in)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'exemplar_size': 127,
                    'score_size': 19, 'ratios': [0.33, 0.5, 1.0, 2.0, 3.0], 'scales': [8.0], 'instance_size': 271,
                    'adaptive': False}  # 0.655


class SiamRPNbatchOTBMobileFeature(SiamRPNbatchOTBMobile):
    def __init__(self):
        super(SiamRPNbatchOTBMobileFeature, self).__init__()

    def forward(self, x):
        x_f = self.featureExtract(x)

        cinput = self.conv_cls2(x_f)
        rinput = self.conv_r2(x_f)

        return cinput, rinput


class SiamRPNbatchOTBMobileTemplate(SiamRPNbatchOTBMobile):
    def __init__(self):
        super(SiamRPNbatchOTBMobileTemplate, self).__init__()

    def forward(self, z):
        z_f = self.featureExtract(z)
        c_kernel_raw = self.conv_cls1(z_f)
        r_kernel_raw = self.conv_r1(z_f)
        kernel_size = r_kernel_raw.data.size()[-1]
        r_kernel = r_kernel_raw.view(self.anchor * 4, self.feature_out, kernel_size, kernel_size)
        c_kernel = c_kernel_raw.view(self.anchor * 2, self.feature_out, kernel_size, kernel_size)
        return c_kernel, r_kernel


class SiamRPNBatchOTBMobileConfig(SiamRPNbatchMobile):
    """ Mobile SiamRPN network. Used for NAS.
    """

    def __init__(self, configs, feature_out=16, teacher_in=256):
        if configs is None:
            configs = [3, 8, 16, 16, 384, 32]
        super(SiamRPNBatchOTBMobileConfig, self).__init__(size=1, feature_out=feature_out,
                                                          anchor=5, configs=configs, teacher_in=teacher_in)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'exemplar_size': 127,
                    'score_size': 19, 'ratios': [0.33, 0.5, 1.0, 2.0, 3.0], 'scales': [8.0], 'instance_size': 271,
                    'adaptive': False}  # 0.655

    def new_net_with_scale(self, scales):
        scales = [int(x) for x in scales]
        config = [3] + scales[:5]
        feature_out = scales[-1]
        return SiamRPNBatchOTBMobileConfig(config, feature_out)


class SiamRPNBatchOTBMobileBig(SiamRPNbatchMobile):
    def __init__(self, teacher_in=256):
        configs = [3, 64, 128, 256, 384, 128]
        super(SiamRPNBatchOTBMobileBig, self).__init__(size=1, feature_out=128, anchor=5, configs=configs,
                                                       teacher_in=teacher_in)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'exemplar_size': 127,
                    'score_size': 19, 'ratios': [0.33, 0.5, 1.0, 2.0, 3.0], 'scales': [8.0],
                    'instance_size': 271, 'adaptive': False}  # 0.655


class SharedNode(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pool_stride=1):
        super(SharedNode, self).__init__()
        self.out_convs = nn.ModuleList()
        for out_c in out_channels:
            in_convs = nn.ModuleList()
            for in_c in in_channels:
                op = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride),
                    nn.BatchNorm2d(out_c),
                    nn.MaxPool2d(kernel_size=3, stride=pool_stride),
                    nn.ReLU(inplace=True),
                )
                in_convs.append(op)

            self.out_convs.append(in_convs)

    def forward(self, x_list, weights, offset):
        out_list = []
        for i, in_convs in enumerate(self.out_convs):
            out = 0
            for j, (op, x) in enumerate(zip(in_convs, x_list)):
                out += op(x) * weights[offset+i][j]

            out_list.append(out)

        return out_list


class SharedFeatureNet(nn.Module):
    def __init__(self, feature_out=32):
        super(SharedFeatureNet, self).__init__()

        channels = [8, 16, 32]
        self.node1 = SharedNode(in_channels=[3], out_channels=channels, kernel_size=3, stride=2, pool_stride=2)
        self.node2 = SharedNode(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, pool_stride=2)
        self.node3 = SharedNode(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, pool_stride=1)
        self.node4 = SharedNode(in_channels=channels, out_channels=[feature_out], kernel_size=3, stride=1, pool_stride=1)

        self.weights = torch.autograd.Variable(torch.randn(3+3+1, 3), requires_grad=True)

    def to(self, device):
        self.weights = self.weights.to(device)
        return super(SharedFeatureNet, self).to(device)

    def parameters(self, recurse=True):
        params = super(SharedFeatureNet, self).parameters()
        return params + [self.weights]

    def forward(self, x):
        weights = F.softmax(self.weights, dim=-1)

        outs = self.node1([x], [[1.0], [1.0], [1.0]], 0)
        outs = self.node2(outs, weights, 0)
        outs = self.node3(outs, weights, 3)
        outs = self.node4(outs, weights, 6)
        return outs[0]


class SharedMobileSiamRPN(SiamRPNbatchMobile):
    def __init__(self, feature_out=32, anchor=5, teacher_in=256):
        configs = [3, 8, 16, 16, 384, 32]
        super(SharedMobileSiamRPN, self).__init__(size=1, feature_out=feature_out,
                                                  anchor=anchor, configs=configs, teacher_in=teacher_in)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'exemplar_size': 127,
                    'score_size': 19, 'ratios': [0.33, 0.5, 1.0, 2.0, 3.0], 'scales': [8.0], 'instance_size': 271,
                    'adaptive': False}  # 0.655

        self.featureExtract = SharedFeatureNet(feature_out=32)

    def get_arch_weights(self):
        return self.featureExtract.weights

    def to(self, device):
        self.featureExtract = self.featureExtract.to(device)
        return super(SharedMobileSiamRPN, self).to(device)


if __name__ == '__main__':
    import numpy as np

    model = SharedMobileSiamRPN()

    param_size = np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6
    print('Total parameter size {} M'.format(param_size))

    template = torch.ones((1, 3, 127, 127))
    detection = torch.ones((1, 3, 271, 271))

    model.temple(template)
    y1, y2, _ = model(detection)
    print(y1.shape)  # [1445, 2]
    print(y2.shape)  # [1445, 4]
