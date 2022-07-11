import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import imageio
import numpy as np

from google.colab import drive
drive.mount("/content/drive")

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
                ])
to_image = transforms.ToPILImage()
trainset = MNIST(root='/content/drive/MyDrive/', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator,self).__init__()
    self.in_feats = 784
    self.out_feats = 1
    self.fc0 = nn.Sequential(
        nn.Linear(self.in_feats,512),
        nn.ReLU()
    )
    self.fc1 = nn.Sequential(
        nn.Linear(512,256),
        nn.ReLU()
    )

    self.fc2 = nn.Sequential(
        nn.Linear(256,128),
        nn.ReLU()
    )
     
    self.fc3 = nn.Sequential(
        nn.Linear(128,64),
        nn.ReLU()
    )

    self.fc4 = nn.Sequential(
        nn.Linear(64,self.out_feats),
        nn.Sigmoid()
    )
  def forward(self,x):
    x = x.view(-1, 784)
    x = self.fc0(x)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.fc4(x)
    return x

generator = Generator()
discriminator = Discriminator()
generator.to(device)
discriminator.to(device)


def noisydata(n,feats=128):
  return Variable(torch.randn(n,feats)).to(device)
def highestvalues(n):
  data = Variable(torch.ones(n,1))
  return data.to(device)
def lowestvalues(n):
  data = Variable(torch.zeros(n,1))
  return data.to(device)

gen_optim = optim.Adam(generator.parameters(), lr=2e-4)
dis_optim = optim.Adam(discriminator.parameters(), lr=2e-4)

g_losses = []
d_losses = []
images = []

criteria = nn.BCELoss()


def discriminator_training(real_data,fake_data,optimizer):
  '''
  Steps:
  1. take in real and fake data
  2. compare real data to ones and fake data to zeros to differentiate between them (BCE Loss formula)
  3. Compute errors and then step the optimizer 

  '''
  size = real_data.size(0)
  optimizer.zero_grad()
  

  real_predict = discriminator(real_data)
  real_error = criteria(real_predict,highestvalues(size))

  real_error.backward()

  fake_predict = discriminator(fake_data)
  fake_error = criteria(fake_predict,lowestvalues(size))

  fake_error.backward()
  optimizer.step()

  return real_error + fake_error

def generator_training(fake_data,optimizer):
  '''
  1. main idea is maximizing the d(g(x)) on fake data.

  '''
  size = fake_data.size(0)

  optimizer.zero_grad()
  
  fake_predict = discriminator(fake_data)
  
  fake_error = criteria(fake_predict,highestvalues(size))
  
  fake_error.backward()
  optimizer.step()

  return fake_error


epochs = 100
k = 1
test_image_noise = noisydata(64)
g_losses = []
d_losses = []

generator.train()
discriminator.train()
for epoch in range(epochs):
    g_error = 0.0
    d_error = 0.0
    for i, data in enumerate(trainloader):
        imgs, _ = data
        n = len(imgs)
        for j in range(k):
            fake_data = generator(noisydata(n)).detach()
            real_data = imgs.to(device)
            #d_error += train_discriminator(d_optim, real_data, fake_data)
            d_error  = d_error + discriminator_training(real_data, fake_data,dis_optim)
        fake_data = generator(noisydata(n))
        g_error  = g_error + generator_training(fake_data,gen_optim)
    print(i)
    img = generator(test_image_noise).cpu().detach()
    img = make_grid(img)
    images.append(img)
    g_losses.append(g_error/i)
    d_losses.append(d_error/i)
    print('Epoch Number: {},  generator"s loss: {:.8f} discriminator"s loss: {:.8f}\r'.format(epoch, g_error/i, d_error/i))

print("Training is done!")
torch.save(generator.state_dict(), 'generator.pth')
