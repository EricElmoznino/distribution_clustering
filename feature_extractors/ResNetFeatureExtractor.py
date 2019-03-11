import torch
from torch import nn
from torchvision.models.resnet import resnet152
from torchvision.transforms import functional as tr
from PIL import Image
from feature_extractors.BaseFeatureExtractor import BaseFeatureExtractor


class ResNetFeatureExtractor(BaseFeatureExtractor):

    def __init__(self):
        self.model = ResNet()
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def get_features(self, sample_path):
        img = Image.open(sample_path).convert('RGB')
        img = tr.resize(img, 224)
        img = tr.to_tensor(img)
        img = tr.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            feats = self.model(img.unsqueeze(dim=0)).squeeze(dim=0)
            feats = feats.cpu().numpy()
        return feats


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = resnet152(pretrained=True)
        resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.features(x).view(x.shape[0], -1)
