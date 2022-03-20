import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

batch_size = 32  # 批次大小
lr = 0.0003
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 读取数据
dataset_train = datasets.ImageFolder(r'E:\学习\深度学习\深度学习\课设\dogs-vs-cats-redux-kernels-edition\train', transform)
dataset_val = datasets.ImageFolder(r'E:\学习\深度学习\深度学习\课设\dogs-vs-cats-redux-kernels-edition\val', transform)
dataset_test = datasets.ImageFolder(r'E:\学习\深度学习\深度学习\课设\dogs-vs-cats-redux-kernels-edition\test', transform)

# 装载数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True)

class Inception(nn.Module):
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.p1 = nn.Sequential(  # 线路1 1*1卷积层
            nn.Conv2d(in_c, c1, kernel_size=1), nn.ReLU())
        self.p2 = nn.Sequential(  # 线路2 1*1卷积层接3*3卷积层
            nn.Conv2d(in_c, c2[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1), nn.ReLU())
        self.p3 = nn.Sequential(  # 线路3 1*1卷积层接5*5卷积层
            nn.Conv2d(in_c, c3[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2), nn.ReLU())
        self.p4 = nn.Sequential(  # 线路4 3*3最大池化层后接1*1卷积层
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), nn.Conv2d(in_c, c4, kernel_size=1), nn.ReLU())

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出

class InceptionModel(nn.Module):
    def __init__(self):
        super(InceptionModel, self).__init__()
        self.b = nn.Sequential(
            Inception(16, 4, (16, 4), (16, 4), 4),
            Inception(16, 16, (16, 16), (16, 16), 16),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        x = self.b(x)
        x = x.view(-1, 64 * 25 * 25)
        return x

# 创建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.inception = InceptionModel()
        self.fc1 = nn.Linear(64 * 25 * 25, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.inception(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.to(device)  # 使用GPU进行训练
cirterion = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr)  # 定义优化器，使用Adam优化器

def train():
    running_loss = 0.0 # 损失
    train_correct = 0 # 训练正确的个数
    train_total = 0 #总的训练个数
    net.train() # 训练
    for i, data in enumerate(train_loader):
        inputs, train_labels = data  # 获得数据和对应的标签
        # 使用GPU进行加速
        inputs = inputs.to(device)
        labels = train_labels.to(device)
        optimizer.zero_grad()  # 梯度清0
        outputs = net(inputs).to(device)  # 获得模型预测结果
        _, train_predicted = torch.max(outputs, 1)  # 获得最大值，以及最大值所在的位置
        train_correct += (train_predicted == labels).sum()  # 分类正确的个数
        loss = cirterion(outputs, labels)  # 使用损失函数计算损失
        loss.backward()  # 计算梯度
        optimizer.step()  # 修改权值
        running_loss += loss.item() # 损失加起来，求平均损失用
        train_total += train_labels.size(0) # 总个数
    # 输出内容，epoch数，损失，正确的个数，总个数，准确率
    print('train:', epoch + 1, 'epoch', ' loss:', '%.4f ' % (running_loss / train_total), ' accrary:',
          int(train_correct),
          '|', len(train_loader.dataset), ' %.4f ' % (train_correct / train_total))

    # 模型验证
    correct = 0
    val_loss = 0.0
    val_total = 0
    net.eval()
    for data in val_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = cirterion(outputs, labels)
        val_loss += loss.item()
        val_total += labels.size(0)
        correct += (predicted == labels.data).sum()
    print('val:', epoch + 1, 'epoch,', 'loss:', '%.4f ' % (val_loss / val_total), 'accrary:', int(correct),
          '|', len(val_loader.dataset), ' %.4f ' % (correct / val_total))

def test():
    correct = 0
    test_loss = 0.0
    test_total = 0
    net.eval()
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = cirterion(outputs, labels)
        test_loss += loss.item()
        test_total += labels.size(0)
        correct += (predicted == labels.data).sum()
    print('\ntest:', 'loss:', '%.4f ' % (test_loss / test_total), ' accrary:', int(correct),
          '|', len(test_loader.dataset), ' %.4f ' % (correct / test_total))
    labels = labels.cpu().tolist()
    predicted = predicted.cpu().tolist()
    sn.heatmap(confusion_matrix(labels, predicted), annot=True)
    plt.show()

for epoch in range(0, 10):
    train()

test()
