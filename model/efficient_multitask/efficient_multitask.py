from model.efficient_multitask.efficient_multitask_utils import EfficientNet
import torch
from torch import nn


class LevelAttentionModel(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(LevelAttentionModel, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(feature_size, 1, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.conv5(out)
        out_attention = self.output_act(out)
        return out_attention


class PyramidFeatures(nn.Module):
    def __init__(self, x0_size, x1_size, x2_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.x2_conv = nn.Conv2d(x2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.x2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        self.x1_conv = nn.Conv2d(x1_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.x1_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.x0_conv = nn.Conv2d(x0_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.x0_out = nn.ReLU()

    def forward(self, inputs):
        x0, x1, x2 = inputs
        x0, x1, x2 = self.x0_conv(x0), self.x1_conv(x1), self.x2_conv(x2)
        x2 = self.x2_upsampled(x2)

        x1 = x1 + x2
        x1 = self.x1_upsampled(x1)

        x0 = x1 + x0
        x0 = self.x0_out(x0)

        return x0


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=2, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 2)

    def forward(self, x, bs, vs):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.avgpool(out)
        out = torch.mean(out.view(bs, vs, -1), dim=1)
        # out = F.normalize(out, p=2, dim=-1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class Efficient_Attention(nn.Module):
    def __init__(self):
        super(Efficient_Attention, self).__init__()
        self._extract_features = EfficientNet.from_pretrained(model_name='efficientnet-b4', num_classes=2)
        self._attention = LevelAttentionModel(num_features_in=256)
        self._feature_pyramid = PyramidFeatures(56, 160, 448)
        self._classification = ClassificationModel(256, num_classes=2)

    def forward(self, x):
        bs, vs = x.size()[0], x.size()[1]
        x = x.view(bs * vs, 3, 256, 256)
        x0, x1, x2 = self._extract_features(x)
        feature = self._feature_pyramid([x0, x1, x2])
        attention = self._attention(feature)

        feature = feature * torch.exp(attention)
        classification = self._classification(feature, bs, vs)
        return classification, attention


def get_efficient_attention():
    net = Efficient_Attention()
    net = nn.DataParallel(net).cuda()
    # for param in net.module._extract_features.parameters():
    #     param.requires_grad = False
    return net
