import time
import copy
import torch
import torch.nn as nn  # 神经网络层
import torch.utils.data as data  # 数据处理
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from torchvision import transforms, datasets
from model import AlexNet

EPOCH = 300  # 迭代次数
BATCH_SIZE = 32  # 批次大小
LR = 0.0001  # 学习率
TRAIN_ROOT = "./MyIMAGE"
IMAGE_SIZE = [227, 227]  # (96,112) 227


def train_data_deal(train_root):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(IMAGE_SIZE, antialias=True),
                                    transforms.RandomHorizontalFlip()])  # 数据集随机水平翻转
    train_val_data = datasets.ImageFolder(train_root, transform)  # 将路径中数据 整理为数据集
    print(train_val_data.class_to_idx)  # 输出类与id之间关系
    return train_val_data


def train_data_shuffle(train_val_data):  # (输入文件位置)
    train_data, val_data = data.random_split(train_val_data,  # 把数据集分成：8训练集2验证集，设置每次训练数据个数，随机打乱训练集。
                                             [round(len(train_val_data) * 0.8), round(len(train_val_data) * 0.2)])
    train_loader = data.DataLoader(dataset=train_data,  # 训练集数据（80%）
                                   batch_size=BATCH_SIZE,  # 一捆的数量
                                   shuffle=True)  # 随机打乱
    val_loader = data.DataLoader(dataset=val_data,  # 验证集数据（20%）
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)
    return train_loader, val_loader


def train_model(model, train_data_loader, val_loader=None):  # 两参数则为每次迭代交叉训练，三参数则为固定数据训练
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 训练器
    criterion = nn.CrossEntropyLoss()  # 损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 放入设备（？）

    best_acc = 0.0  # 初始化最高精确度
    train_loss_all = []  # 训练集损失函数列表
    train_acc_all = []  # 训练集准确度表
    val_loss_all = []
    val_acc_all = []
    a = 0  # 不为交叉训练
    for epoch in range(EPOCH):  # 每一迭代
        since = time.time()  # 计时
        if (val_loader is None) or (a == 1):
            train_loader, val_loader = train_data_shuffle(train_data_loader)
            a = 1
        else:
            train_loader = train_data_loader

        train_loss = 0.0
        train_acc = 0
        val_loss = 0.0
        val_acc = 0
        train_num = 0
        val_num = 0
        for step, (x, y) in enumerate(train_loader):  # 每一捆
            b_x, b_y = Variable(x), Variable(y)  # 设为变量
            b_x, b_y = b_x.to(device), b_y.to(device)  # 放入设备
            # print(b_x.size())  # 可查看数据维度，以修改神经网络层
            model.train()
            output = model(b_x)  # 前向传播
            loss = criterion(output, b_y)  # 计算损失值
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新

            _, pre_y = torch.max(output.data, 1)  # 每个batch最大值索引（[0]为最大值）
            train_loss += loss.data*len(x)  # 累加每批次的每训练集loss
            train_acc += torch.sum(pre_y == b_y)
            train_num += len(x)
            # if step % 50 == 0 and step != 0:  # 每50批数据的平均每批数据
            #     print("now train step:{}, loss:{:.4f}, accuracy:{}"
            #           .format(step, train_loss/train_num, train_acc/train_num))
        for step, (x, y) in enumerate(val_loader):
            b_x, b_y = Variable(x), Variable(y)
            b_x, b_y = b_x.to(device), b_y.to(device)
            model.eval()
            output = model(b_x)
            loss = criterion(output, b_y)
            pre_y = torch.max(output, 1)[1]
            print(pre_y, b_y)
            val_loss += loss.item() * len(x)
            val_acc += torch.sum(pre_y == b_y)
            val_num += len(x)
            # if step % 50 == 0 and step != 0:
            #     print("now val step:{}, loss:{:.4f}, accuracy:{}".format(step, val_loss/val_num, val_acc/val_num))
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_acc/train_num)
        val_acc_all.append(val_acc/val_num)
        print('epoch:{}, Train Loss:{:.4}, Train Acc:{:.4f}，Val Loss:{:.4}, Val Acc:{:.4f}, Consumed Time：{:.4f}'
              .format(epoch+1, train_loss_all[-1], train_acc_all[-1],
                      val_loss_all[-1], val_acc_all[-1], time.time()-since))
        print("-" * 20)
        if val_acc_all[-1] >= best_acc and train_acc_all[-1] >= best_acc:  # 精确度高则保存参数
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())  # 保存参数
            torch.save(best_model_wts, 'best_model_alex.pth')  # 参数保存至路径
    train_process = pd.DataFrame(data={"epoch": range(EPOCH),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})  # 记录训练过程数据
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()


if __name__ == "__main__":
    train_data1 = train_data_deal(TRAIN_ROOT)
    # train_data1, val_data1 = train_data_deal(train_data1)
    Net1 = AlexNet()
    # Net1.load_state_dict(torch.load('best_model_alex.pth'))
    train_process1 = train_model(Net1, train_data1, val_loader=None)
    # matplot_acc_loss(train_process1)