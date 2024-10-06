import logging
from re import S
import torch
import os
import time
import numpy as np
from torch import nn, optim
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler

stamp = time.strftime("%d%H%M%S")
out_dir = os.path.join('./out',stamp)

BATCH_SIZE = 60
SAVE_FREQUENCE=1
SPLIT={
        'nailong':'./data/images/nailong', 'other':'./data/images/bqb' 
        }

device = 'cuda'


def imgLoader(path):
    return Image.open(path).convert('RGB')



train_dataset = ImageFolder('./data/train', loader=imgLoader, transform=transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ToTensor()
    ]))

vali_dataset = ImageFolder('./data/test', loader=imgLoader, transform=transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ToTensor()
    ]))

train_loader = DataLoader(dataset=train_dataset, sampler=ImbalancedDatasetSampler(train_dataset) ,batch_size=BATCH_SIZE)
vali_loader = DataLoader(dataset=vali_dataset,batch_size=BATCH_SIZE)

# print(len(train_dataset),len(test_dataset))
model = torchvision.models.resnet50(pretrained=True)

# 替换最后一层
num_ftrs = model.fc.in_features  
model.fc = nn.Linear(num_ftrs, 2)

model = model.cuda()

 # ------------------------------------ step3: optimizer, lr scheduler ------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-3)

print(f'Start training')  
os.mkdir(out_dir)

# img,idx = nailong_dataset.__getitem__(5)
# print(img)
# plt.imshow(img)
# plt.show()

# train_data, train_type = next(iter(train_loader))

# print(train_data,train_type)

for epoch in range(0,100):
    model.train()
    train_idx = 0

    train_loss_total = 0
    train_acc_total = 0

    for data, labels in tqdm(train_loader):
        train_idx+=1

        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(dim=1)
        acc  = (pred == labels).float().mean()
        # print('label',[torch.sum(labels.eq(i)).item() for i in range(2)])
        # print('pred',[torch.sum(pred.eq(i)).item() for i in range(2)])

        # print(loss.item(),acc.item())

        train_loss_total += loss.item()
        train_acc_total += acc

    model.eval()
    vali_idx = 0

    vali_loss_total = 0
    vali_acc_total = 0

    with torch.no_grad():
        for data, labels in tqdm(vali_loader):
            vali_idx+=1

            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            pred = outputs.argmax(dim=1)
            acc  = (pred == labels).float().mean()

            vali_loss_total += loss.item()
            vali_acc_total += acc

    print(
    f'''Epochs: {epoch + 1} | 
        Loss: {train_loss_total / train_idx: .8f} | 
        Accuracy: {train_acc_total / train_idx: .8f} |
        val_Accuracy: {vali_acc_total / vali_idx: .8f} |
       ''')
    if (SAVE_FREQUENCE and not epoch % SAVE_FREQUENCE):
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f'e{epoch+1}.pt'))
