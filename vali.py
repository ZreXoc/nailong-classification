from PIL import Image
import torch
import sys
import os

import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchsampler.imbalanced import torchvision
from torchvision import transforms
from tqdm import tqdm, trange

model_path='./out/1103-011540/model_best.pth'

batch_size = 10

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img_transform =  transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ])

img_paths = sys.argv[1:]

class ImageDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        try:
            img = Image.open(img_paths[index]).convert('RGB')
            if(self.transform): img = self.transform(img)
            return img_paths[index], img

        except (OSError) as e:
            print(f"{img_paths[i]}: {e}")
            return None

    def __len__(self):
        return len(self.paths)


image_dataset= ImageDataset(img_paths, img_transform)
image_dataloader = DataLoader(image_dataset,batch_size=batch_size,shuffle=False,drop_last=False)

model = torchvision.models.resnet50()
num_ftrs = model.fc.in_features  
model.fc = torch.nn.Linear(num_ftrs, 2)

load = torch.load(model_path,weights_only=True)
state_dict = load['model']
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()


for paths, images in image_dataloader:
    # print(images)
    images = images.to(device)
    outputs = model(images)
    pred = outputs.argmax(dim=1)
    prob = nn.Softmax(dim=1)(outputs)
    isNailong = prob[:,1] > 0.70

    for i in range(len(paths)):
        print(f'{os.path.basename(paths[i])}: {prob[i][1]*100:.1f}%, {"y" if isNailong[i] else "n"}')

# for i in range(len(img_paths)):
    # image = images[i].unsqueeze(0)
    # image = image.cuda()
    # outputs = model(image)

    # pred = outputs.argmax(dim=1)

    # prob = nn.Softmax(dim=1)(outputs)[0]

    # X.append(prob[1].item())

    # isNailong = prob[1] > 0.70

    # print(f'{os.path.basename(img_paths[i])}: {prob[1]*100:.1f}%, {"y" if isNailong else "n"}')

# n, bins, patches = plt.hist(X,bins=40,density=True,cumulative=True)
# plt.plot(bins[:-1], n, linestyle = '--', lw = 2, color = 'r')
# plt.show()
