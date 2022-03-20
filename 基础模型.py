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

batch_size = 64  # 批次大小
lr = 0.0003
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize([224, 224]),  # 图像定义的尺寸要仔细设置，这会影响cuda的利用率，影响运行速度
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

# 创建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 49 * 49, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 16 * 49 * 49)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.to(device)  # 使用GPU进行训练
cirterion = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr)  # 定义优化器，使用Adam优化器

def train():
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    net.train()
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
        running_loss += loss.item()
        train_total += train_labels.size(0)
    print('train:', epoch + 1, 'epoch', ' loss:', '%.4f ' % (running_loss / train_total), ' accrary:',
          int(train_correct),
          '|', len(train_loader.dataset), ' %.4f ' % (train_correct / train_total)) # 输出信息

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
    print('val:', epoch + 1, 'epoch', ' loss:', '%.4f ' % (val_loss / val_total), ' accrary:', int(correct),
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
