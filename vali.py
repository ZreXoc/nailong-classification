import glob
import importlib
from PIL import Image
import re
import torch
import sys
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # !=cuda support
import proccess_gif
from safetensors.torch import load_file
from torch import nn
from torchsampler.imbalanced import torchvision
from torchvision import transforms


def is_gif(file_path):
    try:
        with Image.open(file_path) as img:
            if img.format == 'GIF':
                return True
    except IOError:
        pass
    return False


img_trans =  transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ToTensor()
    ])

img_paths = sys.argv[1:]

images = []
max_number = 0

model = torchvision.models.resnet50()
num_ftrs = model.fc.in_features  
model.fc = torch.nn.Linear(num_ftrs, 2)
folder_path = './out'

# 使用glob模块获取所有e*.txt文件
files = glob.glob(os.path.join(folder_path, 'e*.safetensors'))
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
        model.load_state_dict(load_file(max_file))
else:
    print("No files found matching the pattern.")
    raise FileNotFoundError("NO MODEL FOUND")
    # print(len(train_dataset),len(test_dataset))
model = model.cuda()
model.eval()

X = []
y = []
def de(img_paths):
    needr = False
    for i in range(len(img_paths)):
        if is_gif(img_paths[i]):
            needr = True
            r,d = proccess_gif.process(img_paths[i],'test',output_folder='./')
            img_paths = []
            for l in range(d):
                img_paths = r
            break
    i = 0
    t = 0
    print(img_paths)
    for n in img_paths:
            img = Image.open(n).convert('RGB')
            img = img_trans(img)
            img = img.unsqueeze(0)
            nl = img.cuda()
            outputs = model(nl)
            prob = nn.Softmax(dim=1)(outputs)[0]
            X.append(prob[1].item())
            isNailong = prob[1] > 0.90
            t = float("{:.1f}".format(prob[1]*100)) + t
            print(f'{n}: {prob[1]*100:.1f}%, {"y" if isNailong else "n"}')
            i = i + 1
            if needr:
                os.remove(n)
    return t / i

print(de(img_paths))
