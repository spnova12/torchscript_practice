import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    """
    custom weights initialization called on netG and netD
    https://github.com/pytorch/examples/blob/master/dcgan/main.py
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """
    def __init__(self, nChannels, nDenselayer, growthRate):
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        # local residual 구조
        out = out + x
        return out


def RDB_Blocks(channels, size):
    bundle = []
    for i in range(size):
        bundle.append(RDB(channels, nDenselayer=8, growthRate=64))  # RDB(input channels,
    return nn.Sequential(*bundle)


class Generator_one2many_RDB(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB, self).__init__()

        self.layer1 = nn.Conv2d(input_channel, 64,  kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64,  kernel_size=4, stride=2, padding=1)

        #self.layer4 = ResidualBlocks(64, 15)
        self.layer4 = RDB_Blocks(64, 16)

        self.layer7 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return torch.tanh(out) + x


class Generator_one2many_RDB_no_tanh(nn.Module):
    def __init__(self, input_channel):
        super(Generator_one2many_RDB_no_tanh, self).__init__()

        self.layer1 = nn.Conv2d(input_channel, 64,  kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(64, 64,  kernel_size=4, stride=2, padding=1)

        self.layer4 = RDB_Blocks(64, 16)

        self.layer5 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        # global residual 구조
        return out + x



class DCAD(nn.Module):
    """
    DCAD 모델
    """
    def __init__(self, input_channel):
        super(DCAD, self).__init__()
        num_of_layers = 10
        layers = []
        layers.append(nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, padding=1))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=input_channel, kernel_size=3, padding=1))
        self.dcad = nn.Sequential(*layers)

    def forward(self, x):
        res = self.dcad(x)
        out = x + res
        return out