import torch
from PIL import Image
from lenet import LeNet
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))

im = Image.open('5.jpg')
show_img = im
im = transform(im)  # [C, H, W]
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
im = im.to(device)

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].numpy()
print(classes[int(predict)])

pred = net(im)
print(pred.argmax(dim = 1).cpu().numpy()[0])
res = ''
res += classes[pred.argmax(dim = 1)]
plt.figure("Predict")
plt.imshow(show_img)
plt.axis("off")
plt.title('pred:' +res)
plt.show()

