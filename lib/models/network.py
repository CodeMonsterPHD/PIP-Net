import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
from lib.utils.utils import cosine
from lib.config.para import parse_opt
opt = parse_opt()

class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inplanes = 64
        block, n_blocks, channels = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, n_blocks[0], channels[0])
        self.layer2 = self._make_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self._make_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self._make_layer(block, n_blocks[3], channels[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, blocks , planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_center(self,center):
        self.center = center
    def forward(self, x, state='None'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        fea_without_nor = x
        if state == 'train' or state == 'test':
            # for train and test
            x = x.squeeze(dim=3).squeeze(dim=2)
            # normalize the feature
            x = torch.div(x,torch.sum(x,dim=1).view(-1,1))
            c = self.center
            c = c.type_as(x).cuda()
            dis = cosine(x, c)
            dis1 = dis.view(len(dis),opt.Intent_class,-1)
            pre, _ = torch.max(dis1,dim=2)
            pre = pre/opt.temperature
            return pre
        elif state == "get":
            # get the feature.
            return x
        else:
        # for CAM
            x = fea_without_nor
            # normalize the feature
            x = torch.div(x, torch.sum(x, dim=1).view(-1, 1))
            c = self.center
            c = c.type_as(x).cuda()
            x = x.squeeze(dim=3).squeeze(dim=2)
            dis = cosine(x, c)
            dis = dis.view(len(dis),opt.Intent_class,-1)
            pre, _ = torch.max(dis,dim=2)
            pre = pre/opt.temperature
            softmax = torch.nn.Softmax(dim=1)
            pre = softmax(pre)
            return pre

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x




if __name__ == '__main__':

    # ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    # output_dim = 28
    # resnet50_config = ResNetConfig(block=Bottleneck,
    #                                n_blocks=[3, 4, 6, 3],
    #                                channels=[64, 128, 256, 512])
    # model = ResNet(resnet50_config, output_dim)
    #
    # data = torch.randn((2, 3, 224, 224))
    # out = model(data)
    # print(out.size())

    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

    resnet50_config = ResNetConfig(block=Bottleneck,
                                   n_blocks=[3, 4, 6, 3],
                                   channels=[64, 128, 256, 512])
    pretrained_model = models.resnet50(pretrained=True)
    # print(pretrained_model)

    IN_FEATURES = pretrained_model.fc.in_features
    output_dim = 28  # 200
    fc = nn.Linear(IN_FEATURES, output_dim)
    pretrained_model.fc = fc

    model = ResNet(resnet50_config, output_dim)
    model.load_state_dict(pretrained_model.state_dict(),False)

    data = torch.randn((2, 3, 224, 224))
    out = model(data)
    print(out.size())

