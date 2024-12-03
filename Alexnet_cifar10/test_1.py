import torch
from torchvision import datasets
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision.transforms as transforms
from alexnet import AlexNet

transform = transforms.Compose([
    transforms.Resize([32,32]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
    ])

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset,
                         batch_size = 256,
                         shuffle = False,
                         )

def predict():
    net = AlexNet().to(device)
    net.load_state_dict(torch.load('./Alexnet.pth'))
    print(net)
    total_correct = 0
    for batch_idx, (x, y) in enumerate(test_loader):
        # x = x.view(x.size(0),28*28)
        # x = x.view(256,28,28)
        x = x.to(device)
        # print(x.shape)
        y = y.to(device)
        # print('y',y)
        out = net(x)
        # print(out)
        pred = out.argmax(dim=1)
        # print('pred',pred)
        correct = pred.eq(y).sum().float().item()
        total_correct += correct
    total_num = len(test_loader.dataset)

    acc = total_correct / total_num
    print("test acc:", acc)

predict()
