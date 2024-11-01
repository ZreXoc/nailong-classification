import glob
from re import S
import torch
import os
from safetensors.torch import save_file,load_file
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler
import re
BATCH_SIZE = 55
SAVE_FREQUENCE=1
# SPLIT={
#         'nailong':'./data/images/nailong', 'other':'./data/images/bqb' 
#         }

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
# 定义文件夹路径
folder_path = './out'  # 替换为您的文件夹路径

# 使用glob模块获取所有e*.txt文件
files = glob.glob(os.path.join(folder_path, 'e*.safetensors'))

# 初始化最大数字和对应的文件路径
max_number = 0
max_file = None
model = torchvision.models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 替换最后一层为2类输出

model = model.to(device)

# 遍历文件列表
for file in files:
    # 使用正则表达式提取文件名中的数字
    match = re.search(r'e(\d+)\.safetensors', os.path.basename(file))
    if match:
        number = int(match.group(1))
        # 如果当前数字大于已记录的最大数字，则更新最大数字和对应的文件路径
        if number > max_number:
            max_number = number
            max_file = file
# 打印最大数字对应的文件路径
if max_file:
    print(f"The file is: {max_file} using this model")
    if os.path.isfile(max_file):
        model.load_state_dict(load_file(max_file, device=device))
else:
    print("No files found matching the pattern. creating new model")
    # print(len(train_dataset),len(test_dataset))

    # 替换最后一层
    


# 加载之前保存的模型状态


model = model.cuda()

 # ------------------------------------ step3: optimizer, lr scheduler ------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-3)

print(f'Start training')  
if os.path.exists(folder_path):
    # 如果目标路径存在，检查它是否是一个目录
    if os.path.isdir(folder_path):
        print(f"Directory '{folder_path}' already exists.")
    else:
        # 如果目标路径是一个文件，您可以选择删除它或者抛出错误
        print(f"Error: '{folder_path}' is a file and cannot be overwritten by a directory.")
        # 如果您确定要删除这个文件，请取消注释以下代码
        # os.remove(folder_path)
        # os.makedirs(folder_path)
else:
    # 如果目标路径不存在，创建目录
    os.makedirs(folder_path)

# img,idx = nailong_dataset.__getitem__(5)
# print(img)
# plt.imshow(img)
# plt.show()

# train_data, train_type = next(iter(train_loader))

# print(train_data,train_type)

for epoch in range(0, 60):
    model.train()
    train_idx = 0

    train_loss_total = 0
    train_acc_total = 0

    for data, labels in tqdm(train_loader):
        train_idx += 1

        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(dim=1)
        acc = (pred == labels).float().mean()

        train_loss_total += loss.item()
        train_acc_total += acc

    model.eval()
    vali_idx = 0

    vali_loss_total = 0
    vali_acc_total = 0

    with torch.no_grad():
        for data, labels in tqdm(vali_loader):
            vali_idx += 1

            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            pred = outputs.argmax(dim=1)
            acc = (pred == labels).float().mean()

            vali_loss_total += loss.item()
            vali_acc_total += acc

    print(
        f'''Epochs: {max_number + epoch + 1} | 
        Loss: {train_loss_total / train_idx: .8f} | 
        Accuracy: {train_acc_total / train_idx: .8f} |
        val_Accuracy: {vali_acc_total / vali_idx: .8f} |
        ''')

    if (SAVE_FREQUENCE and not epoch % SAVE_FREQUENCE):
        save_file(
            model.state_dict(),
            os.path.join(folder_path, f'e{max_number+epoch + 1}.safetensors')
        )
