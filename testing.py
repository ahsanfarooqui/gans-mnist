import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self):
      super(Generator,self).__init__()
      self.in_feats = 128
      self.out_feats = 784
      self.fc0 = nn.Sequential(
          nn.Linear(self.in_feats,256),
          nn.ReLU()          
      )
      self.fc1 = nn.Sequential(
          nn.Linear(256,512),
          nn.ReLU()
      )
      self.fc2 = nn.Sequential(
          nn.Linear(512,1024),
          nn.ReLU()
      )
      self.fc4 = nn.Sequential(
          nn.Linear(1024,self.out_feats),
          nn.Tanh()
      )
    def forward(self,x):
      x = self.fc0(x)
      x = self.fc1(x)
      x = self.fc2(x)
      x = self.fc4(x)
      x = x.view(-1,1,28,28)
      return x
    
device = "cpu"
generator  = Generator()
generator.load_state_dict(torch.load("generator.pth", map_location=device))


generator.eval()

def noisydata(n,feats=128):
  return Variable(torch.randn(n,feats)).to(device)

import matplotlib.pyplot as plt
for i in range(10):
    img = generator(noisydata(1))
    plt.imshow(img.detach()[0][0])
    plt.show()



