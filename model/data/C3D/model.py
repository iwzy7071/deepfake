import torch
import torch.nn as nn


class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self._conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self._pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self._conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self._pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self._conv3a = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self._conv3b = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self._pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self._conv4a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self._conv4b = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self._pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self._conv5a = nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self._conv5b = nn.Conv3d(1024, 2048, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self._pool5 = nn.AdaptiveAvgPool3d(1)

        self._fc6 = nn.Linear(2048, 1024)
        self._fc7 = nn.Linear(1024, 2)

        self._dropout = nn.Dropout(p=0.5)

        self._relu = nn.ReLU()
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self._relu(self._conv1(x))
        x = self._pool1(x)

        x = self._relu(self._conv2(x))
        x = self._pool2(x)

        x = self._relu(self._conv3a(x))
        x = self._relu(self._conv3b(x))
        x = self._pool3(x)

        x = self._relu(self._conv4a(x))
        x = self._relu(self._conv4b(x))
        x = self._pool4(x)

        x = self._relu(self._conv5a(x))
        x = self._relu(self._conv5b(x))
        x = self._pool5(x)

        x = x.view(x.size()[0], -1)
        x = self._relu(self._fc6(x))
        x = self._dropout(x)
        logits = self._fc7(x)
        probs = self._softmax(logits)

        return probs


if __name__ == '__main__':
    net = C3D()
    input = torch.rand([32, 3, 8, 299, 299])
    probs = net(input)
    print(probs.size())
