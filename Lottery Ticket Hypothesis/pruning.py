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


class MaskedModel(MResnet):
  def __init__(self, in_channels, num_classes, device, pre_masks=None):
    super().__init__(in_channels, num_classes)
    self.device = device
    if pre_masks is None:
      self._masks = []
    else:
      self._masks = pre_masks
    self._prune_rates = []
    self.init_masks()

  def init_masks(self):
    if self._masks == []:
      for iter, p in enumerate(self.parameters()):
        mask = torch.ones_like(p.data, dtype=float)
        self._masks.append(mask)

    else:
      for iter, p in enumerate(self.parameters()):
        size_match = (self._masks[iter].shape == p.shape)
        assert size_match
  
  def set_masks(self, masks):
    for iter, p in enumerate(self.parameters()):
      size_match = (masks[iter].shape == p.shape)
      assert size_match

    self._masks = masks

    for i in range(len(self._masks)):
      self._masks[i] = to_device(self._masks[i], device)

  def get_masks(self):
    return self._masks
  
  def forward(self, x):
    for iter, p in enumerate(self.parameters()):
      new_p = torch.mul(p,self._masks[iter])
      new_p = new_p.type(torch.cuda.FloatTensor)
      p.data = new_p.data

    return super().forward(x)


def random_pruning(model: MaskedModel, p=0.05):
  for iter, mask in enumerate(model._masks):
    new_mask = torch.ones_like(mask)*(1-p)
    new_mask = torch.bernoulli(new_mask)
    return new_mask


def iterative_magnitude_pruning(model: MaskedModel, prune_rate=0.05):
  parameters = model.parameters()
  l = [torch.flatten(p) for p in parameters]
  flat = torch.abs(torch.cat(l).view(-1, 1))
  flat = flat[flat>1e-9]

  threshold = torch.quantile(flat, prune_rate)
  print(threshold)

  new_mask = []
  for iter, p in enumerate(model.parameters()):
    temp_mask = (torch.abs(p) > threshold).type(torch.cuda.FloatTensor)
    new_mask.append(temp_mask)

  return new_mask


def fit (epochs,train_dl,test_dl,model,optimizer,max_lr,weight_decay,scheduler, prune_rate=0, grad_clip=None, save_warmup=False, warmup_steps=100):
    torch.cuda.empty_cache()
    
    history = []

    step = 0

    warmup_model = None
    
    optimizer = optimizer(model.parameters(),max_lr,weight_decay=weight_decay)
    
    scheduler = scheduler(optimizer,max_lr,epochs=epochs,steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        model.train()
        
        train_loss = []
        
        lrs = []
        
        length = len(train_dl)

        for batch_iter, batch in enumerate(train_dl):
            if step == warmup_steps and save_warmup:
              warmup_model = model.state_dict()

            loss = model.training_step(batch)
            
            train_loss.append(loss)

            print('\r', f"[Epoch {epoch}/{epochs}], [Batch {batch_iter}/{length}] loss: {loss.item():.4f}", end=' ')
            
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(),grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()
            lrs.append(get_lr(optimizer))

            step += 1

        result = evaluate(model,test_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["lrs"] = lrs
        
        model.epoch_end(epoch,result)
        history.append(result)

        

    if save_warmup:
      return history, iterative_magnitude_pruning(model, prune_rate), warmup_model
    
    else:
      return history, iterative_magnitude_pruning(model, prune_rate)
