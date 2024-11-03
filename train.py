import logging
import re
import torch
import os
import time
import numpy as np
from torch import Tensor, nn, optim
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision.datasets import ImageFolder
from PIL import Image,ImageFile
from torchvision.datasets.imagenet import shutil
from torchvision.transforms import transforms
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

stamp = time.strftime("%m%d-%H%M%S")
out_dir = os.path.join('./out',stamp)

BATCH_SIZE = 65
SAVE_FREQUENCE=5

device = 'cuda'


def imgLoader(path):
    return Image.open(path).convert('RGB')



train_dataset = ImageFolder('./data/train', loader=imgLoader, transform=transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ]))

vali_dataset = ImageFolder('./data/test', loader=imgLoader, transform=transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ]))

train_loader = DataLoader(dataset=train_dataset, sampler=ImbalancedDatasetSampler(train_dataset) ,batch_size=BATCH_SIZE)
vali_loader = DataLoader(dataset=vali_dataset,batch_size=BATCH_SIZE)


str_loong = ['nailong', 'nailoong', '奶龙']
is_label_nailong=[]

for _class in train_dataset.classes:
    if any(loong in _class for loong in str_loong):
        is_label_nailong.append(True)
    else:
        is_label_nailong.append(False)

is_label_nailong = torch.Tensor(is_label_nailong)
# print(len(train_dataset),len(test_dataset))
model = torchvision.models.resnet50(pretrained=True)

# 替换最后一层
num_ftrs = model.fc.in_features  
model.fc = nn.Linear(num_ftrs, 2)

model = model.cuda()

 # ------------------------------------ step3: optimizer, lr scheduler ------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = 2e-5)

print(f'Start training')  
os.mkdir(out_dir)

best_acc=-1

for epoch in range(0,100):
    train_idx = 0

    train_loss_total = 0
    train_acc_total = 0

    for data, origin_labels in tqdm(train_loader):
        train_idx+=1
        labels = torch.zeros_like(origin_labels)
        labels[is_label_nailong[origin_labels]==True] = 1

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
        for data, origin_labels in tqdm(vali_loader):
            vali_idx+=1

            labels = torch.zeros_like(origin_labels)
            labels[is_label_nailong[origin_labels]==True] = 1

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
    val_acc = vali_acc_total /vali_idx
    path = os.path.join(out_dir, f'e{epoch+1}.pth')
    if(best_acc < val_acc): 
        print(f'better acc {val_acc} than {best_acc}')
        best_acc = val_acc;
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), \
              'epoch': epoch}
        torch.save(checkpoint , path)
        shutil.copyfile(path, os.path.join(out_dir, 'model_best.pth'))

    if (SAVE_FREQUENCE and not epoch % SAVE_FREQUENCE):
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), \
              'epoch': epoch}
        torch.save(checkpoint , path)
