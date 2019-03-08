import torch
from torch import nn
from torchvision.models.resnet import resnet152
from torchvision.transforms import functional as tr
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import argparse


class ResNetFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet152(pretrained=True)
        resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.features(x).view(x.shape[0], -1)


# Parse arguments
parser = argparse.ArgumentParser()
req_grp = parser.add_argument_group('required')
req_grp.add_argument('--data_dir', default=None, help='directory of data to be clustered.')
args = parser.parse_args()

images = os.listdir(os.path.join(args.data_dir, 'images'))
images = [img for img in images if img != '.DS_Store']
images = sorted(images)
images = [os.path.join(args.data_dir, 'images', img) for img in images]

model = ResNetFeatures()
model.eval()
if torch.cuda.is_available():
    model.cuda()

features = []
for img in tqdm(images):
    img = Image.open(img).convert('RGB')
    img = tr.resize(img, 224)
    img = tr.center_crop(img, (224, 224))
    img = tr.to_tensor(img)
    img = tr.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if torch.cuda.is_available():
        img = img.cuda()
    with torch.no_grad():
        out = model(img.unsqueeze(dim=0)).squeeze(dim=0)
    out = out.cpu().numpy()
    features.append(out)

features = np.stack(features)
np.save(os.path.join(args.data_dir, 'features.npy'), features)
