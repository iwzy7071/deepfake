from model.efficientnet.attention_model import EfficientNet
import torch

if __name__ == '__main__':
    image = torch.randn(size=(16, 128, 299, 299)).cuda()
    net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2).cuda()
    pred = net(image)
