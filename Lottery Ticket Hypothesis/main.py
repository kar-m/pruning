import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split,ConcatDataset
from train_setup import *
from pruning import *


epochs = 25
prune_rate = 0.3
optimizer = torch.optim.Adam
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
scheduler = torch.optim.lr_scheduler.OneCycleLR
BATCH_SIZE = 512


stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
train_transform = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.RandomCrop(32,padding=4,padding_mode="reflect"),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

test_transform = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(*stats)
])

train_data = CIFAR10(download=True,root="./data",transform=train_transform)
test_data = CIFAR10(root="./data",train=False,transform=test_transform)

train_dl = DataLoader(train_data,BATCH_SIZE,num_workers=4,pin_memory=True,shuffle=True)
test_dl = DataLoader(test_data,BATCH_SIZE,num_workers=4,pin_memory=True)

device = get_device()
print(device)

train_dl = ToDeviceLoader(train_dl,device)
test_dl = ToDeviceLoader(test_dl,device)

model = None
warmup_state_dict = None
new_masks = None

for cycle in range(10):
  print(cycle)
  if cycle == 0:
    model = MaskedModel(3, 10, device=device)
    model = to_device(model,device)
    for i in range(len(model._masks)):
      model._masks[i] = to_device(model._masks[i], device)
  
  else:
    model = MaskedModel(3, 10, device=device)
    model.load_state_dict(warmup_state_dict)
    model = to_device(model,device)
    model.set_masks(new_masks)
    
  history = [evaluate(model,test_dl)]
  fit_list = fit(epochs=epochs,train_dl=train_dl,test_dl=test_dl,model=model,optimizer=optimizer,max_lr=max_lr,grad_clip=grad_clip,
              weight_decay=weight_decay,scheduler=torch.optim.lr_scheduler.OneCycleLR, prune_rate=prune_rate, save_warmup=(cycle==0))
  history += fit_list[0]

  new_masks = fit_list[1]

  if cycle == 0:
    warmup_state_dict = fit_list[2]

sum = 0
num = 0
for p in model.parameters():
  sum += torch.sum(p.data != 0)
  num += torch.numel(p.data)

print(sum/num)

