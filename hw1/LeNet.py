import torch.optim
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    # 下载与预处理数据集
    trans = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=trans)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=trans)

    # 加载数据集
    trainLoader = DataLoader(train_set, batch_size=64, shuffle=True)
    testLoader = DataLoader(test_set, batch_size=64, shuffle=False)

    # 指定device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型
    lenet = Lenet()
    lenet = lenet.to(device)
    # tensorboard

    writer = SummaryWriter("./logs")
    # 损失函数
    loss_fn = nn.CrossEntropyLoss().to(device)
    # 优化器
    learning_rate = 0.01
    optimizer = torch.optim.SGD(lenet.parameters(), lr=learning_rate, momentum=0.9)

    # 训练参数
    epoch = 10
    train_step = 0
    test_step = 0

    for i in range(epoch):
        print(f"-------第{i + 1}轮训练--------")
        # 训练步骤
        lenet.train()
        for data in trainLoader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = lenet(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step = train_step + 1
            if train_step % 50 == 0:
                print(f"训练步数： {train_step}, Loss：{loss}")
                writer.add_scalar("train_loss", loss, train_step)

        # 测试步骤
        total_test_loss = 0
        total_accuracy = 0
        lenet.eval()
        with torch.no_grad():
            for data in testLoader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = lenet(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss
                total_accuracy += (outputs.argmax(1) == targets).sum()
        print(f"测试集上的总loss: {total_test_loss}")
        print(f"测试集上的整体正确率：{total_accuracy / len(test_set)}")
        writer.add_scalar("test_loss", total_test_loss, i + 1)
        writer.add_scalar("test_accuracy", total_accuracy / len(test_set), i + 1)

    writer.close()

