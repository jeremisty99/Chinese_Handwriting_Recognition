import os
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from ..hwdb import HWDB

# prepare dataset
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, ], [0.229, ])
])
batch_size = 64  # batchsize
lr = 0.1  # 学习率
data_path = r'../dataset'  # 数据集路径
log_path = r'logs/batch_{}_lr_{}'.format(batch_size, lr)  # 日志路径
save_path = r'checkpoints/'  # 模型保存路径
if not os.path.exists(save_path):
    os.mkdir(save_path)
dataset = HWDB(path=data_path, transform=transform)
print("训练集数据:", dataset.train_size)
print("测试集数据:", dataset.test_size)
train_loader, test_loader = dataset.get_loader(batch_size)


# design model using class


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # [1,32,32] -> [6,28,28] -> [6,14,14]
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, stride=1)
        # [6,14,14]-> [16,10,10] -> [16,5,5] = 400
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc_unit = torch.nn.Sequential(
            torch.nn.Linear(400, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 16)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # -1 此处自动算出的是400 多维度的tensor展平成一维
        # print("x.shape", x.shape)
        x = self.fc_unit(x)
        return x


model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch, net, criterion, optimizer, train_loader, scheduler, save_iter=42):
    print("epoch %d 开始训练..." % epoch)
    net.train()
    sum_loss = 0.0
    total = 0
    correct = 0
    max_acc = 0;
    for i, (inputs, labels) in enumerate(train_loader, 0):
        optimizer.zero_grad()  # 梯度清零
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # 取得分最高的那个类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        # 每训练100个batch打印一次平均loss与acc
        if (i + 1) % save_iter == 0:
            batch_loss = sum_loss / save_iter
            # 每跑完一次epoch测试一下准确率
            acc = 100 * correct / total
            print('epoch: %d, batch: %d loss: %.03f, acc: %.04f'
                  % (epoch, i + 1, batch_loss, acc))
            total = 0
            correct = 0
            sum_loss = 0.0
            if acc > max_acc:
                max_acc = acc
                print("save model")
                # 保存模型语句
                torch.save(model.state_dict(), "model.pth")
        scheduler.step()


if __name__ == '__main__':
    epoch_list = []
    acc_list = []
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    for epoch in range(20):
        train(epoch, model, criterion, optimizer, train_loader, scheduler=scheduler)
    #     epoch_list.append(epoch)
    #     acc_list.append(acc)
    #
    # plt.plot(epoch_list, acc_list)
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.show()
