import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from alexnet import AlexNet
from utils_1 import plot_curve
from torch.utils.data import DataLoader
from torchvision import datasets

# 定义使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置超参数
epochs = 50
batch_size = 256
lr = 0.01

transform = transforms.Compose([
    transforms.Resize([32,32]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
    ])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=False,transform=transform)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle = True,)

net = AlexNet(10)
net.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)

train_loss = []
for epoch in range (epochs):
    sum_loss = 0
    for batch_idx,(x,y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        pred = net(x)

        optimizer.zero_grad()
        loss = loss_func(pred,y)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        train_loss.append(loss.item())
        print(["epoch:%d , batch:%d , loss:%.3f" %(epoch,batch_idx,loss.item())])

    save_path = './Alexnet.pth'
    torch.save(net.state_dict(), save_path)

plot_curve(train_loss)