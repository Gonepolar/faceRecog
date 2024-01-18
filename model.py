import torch
import torch.nn as nn  # 神经网络层


class AlexNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=96,
                            kernel_size=11,
                            stride=4),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Conv2d(96, 256, 5, padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Conv2d(256, 384, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 384, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256*6*6, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 5),
            # dim=1是按行softmax——降到（0,1）区间内相当于概率，此处不用softmax因为定义的交叉熵损失函数CrossEntropy包含了softmax
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous().view(-1, 256*6*6)  # 使用.contiguous()防止用多卡训练的时候tensor不连续，即tensor分布在不同的内存或显存中
        x = self.fc(x)
        return x


class CnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(  # 简单层次包装
            nn.Conv2d(in_channels=3,  # 传入的图片是几层的，灰色为1层，RGB为三层 -->(3,96,112)
                      out_channels=16,
                      kernel_size=5,  # 卷积核大小
                      stride=1,  # 卷积核步长
                      padding=2),  # =(kernel_size-1)/2  # 2d代表二维卷积 -->(16,96,112)
            nn.ReLU(),  # 非线性激活层
            nn.MaxPool2d(kernel_size=4))  # 最大池化 --> (16,24,28)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,  # -->(16,24,28)
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),  # -->(32,24,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4))  # --> (32,6,7)
        self.f3 = nn.Linear(32*6*7, 40)  # 注意：这里数据是二维数据

    def forward(self, x):  # -->(batch,1,28,28)
        x = self.conv1(x)  # -->(batch,16,14,14)
        x = self.conv2(x)  # -->(batch,32,7,7)
        x = x.view(x.size(0), -1)  # 扩展展平，将四维数据转为二维数据 -->(batch, 32*7*7)
        x = self.f3(x)  # -->(batch, 10)
        return x