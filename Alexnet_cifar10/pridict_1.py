##读入文件，显示正确分类和预测分类
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from alexnet import AlexNet

transform = transforms.Compose([
    transforms.Resize([32,32]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]),
    ])

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

img = Image.open('5.jpg')
# img = Image.open(file_name).convert("RGB")
show_img = img
img = transform(img)

# print(img)
# print(img.shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img = img.to(device)
img = img.unsqueeze(0)
net = AlexNet().to(device)
net.load_state_dict(torch.load('./Alexnet.pth'))

pred = net(img)
print(pred.argmax(dim = 1).cpu().numpy()[0])
res = ''
res += classes[pred.argmax(dim = 1)]
#labels = classes[int(file_name.split('/')[-1].split('_')[0])]
plt.figure("Predict")
plt.imshow(show_img)
plt.axis("off")
#plt.title('label:' + labels + '  pred:' +res)
plt.title('pred:' +res)
plt.show()
