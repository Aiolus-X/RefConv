import torch
import numpy as np
import math
import time
import random
import os
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm
import logging
from torch.nn.parameter import Parameter
seed = 42

#num_epochs = [0,30,30,30,30]
start_epoch = 1
exp_file = './exp.log'

data_root = '/data/dataset/ImageNet2012'
#data_root = '/data0/imagenet'
batch_size = 512
input_size = 224
mode = 'large'

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s]%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

logger = get_logger(exp_file)


traindir = os.path.join(data_root, 'train')
valdir = os.path.join(data_root, 'val')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)
np.random.seed(seed)
torch.manual_seed(seed)
dtype = np.float32
kwargs = {"num_workers": 16, "pin_memory": True}
train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,**kwargs)
val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])),
        batch_size=batch_size, shuffle=False,**kwargs)

def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        #self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


net = MobileNetV3(mode=mode)
net = nn.DataParallel(net, device_ids=[0,1,2,3])
net.cuda()
#net_dict = torch.load('./Final.pth')
#net.module.load_state_dict(net_dict,strict=False)


logger.info('start training!')

weight_decay = 4e-5
lr = 1e-1
num_epochs=100

train_loss=[]
val_loss=[]
train_acc = []
val_acc = []

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(net.parameters(), momentum=0.9,lr=lr,weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=5e-6, end_factor=1.0, total_iters=5*len(train_loader), last_epoch=-1)
#5-epoch warmup, initial value of 0.1 and cosine annealing for 100 epochs. 
for epoch in range(5):
    net.train()
    start_time = time.time()
    c1=[]
    total=0
    correct1=0
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x = Variable(x)
        y = Variable(y)
        x=x.cuda()
        y=y.cuda()
        output = net(x)
        loss1 = criterion(output,y)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        scheduler.step()
        #lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
        c1.append(loss1.item())
        total += y.size(0)

        _, predicted = torch.max(output.data, 1)
        correct1 += (predicted == y).sum().item()
                        
    train_loss.append(np.mean(c1))#(torch.mean(torch.stack(c1)))
    t1=100 * correct1 / total
    train_acc.append(t1)
    end_time = time.time()
    #print("Epoch {} loss: {} T1_Accuracy: {}% T5_Accuracy: {}% Time costs: {}s".format(epoch + start_epoch, loss_count[-1], t1, t5, end_time - start_time))
    logger.info("Epoch {} loss: {} T1_Accuracy: {}%  Time costs: {}s".format(epoch + start_epoch, train_loss[-1], t1, end_time - start_time))
    #("Epoch {} Accuracy: {}% Time costs: {}s".format(epoch + start_epoch, t1, end_time - start_time))
    
    net.eval()
    with torch.no_grad():
        c2=[]
        total=0
        correct1=0
        for data in val_loader:
            images, labels = data
            images=images.cuda()
            labels=labels.cuda()
            outputs = net(images).cuda()
            loss2 = criterion(outputs,labels)
            c2.append(loss2.item())
            total += labels.size(0)
                
            _, predicted = torch.max(outputs.data, 1)
            correct1 += (predicted == labels).sum().item()
                
        val_loss.append(np.mean(c2))#(torch.mean(torch.stack(c2)))
        t1=100 * correct1 / total
        val_acc.append(t1)
            
    logger.info('Val_Accuracy:{}%'.format(t1))
    
    torch.save(net.module.state_dict(), './CP.pth')


optimizer = torch.optim.SGD(net.parameters(), momentum=0.9,lr=lr,weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100*len(train_loader), eta_min=5e-7, T_mult=1,last_epoch=-1)

for epoch in range(num_epochs):
    net.train()
    start_time = time.time()
    c1=[]
    total=0
    correct1=0
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x = Variable(x)
        y = Variable(y)
        x=x.cuda()
        y=y.cuda()
        output = net(x)
        loss1 = criterion(output,y)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        scheduler.step()
        #lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
        c1.append(loss1.item())
        total += y.size(0)

        _, predicted = torch.max(output.data, 1)
        correct1 += (predicted == y).sum().item()
                        
    train_loss.append(np.mean(c1))#(torch.mean(torch.stack(c1)))
    t1=100 * correct1 / total
    train_acc.append(t1)
    end_time = time.time()
    #print("Epoch {} loss: {} T1_Accuracy: {}% T5_Accuracy: {}% Time costs: {}s".format(epoch + start_epoch, loss_count[-1], t1, t5, end_time - start_time))
    logger.info("Epoch {} loss: {} T1_Accuracy: {}%  Time costs: {}s".format(epoch + start_epoch, train_loss[-1], t1, end_time - start_time))
    #("Epoch {} Accuracy: {}% Time costs: {}s".format(epoch + start_epoch, t1, end_time - start_time))
    
    net.eval()
    with torch.no_grad():
        c2=[]
        total=0
        correct1=0
        for data in val_loader:
            images, labels = data
            images=images.cuda()
            labels=labels.cuda()
            outputs = net(images).cuda()
            loss2 = criterion(outputs,labels)
            c2.append(loss2.item())
            total += labels.size(0)
                
            _, predicted = torch.max(outputs.data, 1)
            correct1 += (predicted == labels).sum().item()
                
        val_loss.append(np.mean(c2))#(torch.mean(torch.stack(c2)))
        t1=100 * correct1 / total
        val_acc.append(t1)
            
    logger.info('Val_Accuracy:{}%'.format(t1))
    
    torch.save(net.module.state_dict(), './CP.pth')

    
logger.info('finish training!')
torch.save(net.module.state_dict(), './Final.pth')

logger.info(max(val_acc))
logger.info(val_acc[-1])

logger.info(train_acc)
logger.info(val_acc)

logger.info(train_loss)
logger.info(val_loss)

