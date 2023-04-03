import torch
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, \
    PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
from torch import nn
# import pytorch_lightning as pl

# batchnorm params
bn_mom = 0.9
bn_eps = 2e-5
# use_global_stats = False

# net_setting params
use_se = True
se_ratio = 4
group_base = 8


class Se_block(Module):
    def __init__(self, num_filter, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)):
        super(Se_block, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        # self.pool1 = AvgPool2d(?)
        self.conv1 = Conv2d(in_channels=num_filter, out_channels=num_filter // se_ratio, kernel_size=kernel_size,
                            stride=stride, padding=padding)
        self.act1 = PReLU(num_filter // se_ratio)
        self.conv2 = Conv2d(in_channels=num_filter // se_ratio, out_channels=num_filter, kernel_size=kernel_size,
                            stride=stride, padding=padding)
        self.act2 = Sigmoid()

    def forward(self, x):
        temp = x
        x = self.pool1(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        return temp * x


class Separable_Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=(1, 1), factor=1, bias=False,
                 bn_dw_out=True, act_dw_out=True, bn_pw_out=True, act_pw_out=True, dilation=1):
        super(Separable_Conv2d, self).__init__()

        assert in_channels % group_base == 0
        self.bn_dw_out = bn_dw_out
        self.act_dw_out = act_dw_out
        self.bn_pw_out = bn_pw_out
        self.act_pw_out = act_pw_out
        # depthwise
        self.dw1 = Conv2d(in_channels=in_channels, out_channels=int(in_channels * factor), kernel_size=kernel_size,
                          stride=stride, padding=padding, dilation=dilation, groups=int(in_channels / group_base),
                          bias=bias)
        if self.bn_dw_out:
            self.dw2 = BatchNorm2d(num_features=int(in_channels * factor), eps=bn_eps, momentum=bn_mom,
                                   track_running_stats=True)
        if act_dw_out:
            self.dw3 = PReLU(int(in_channels * factor))

        # pointwise
        self.pw1 = Conv2d(in_channels=int(in_channels * factor), out_channels=out_channels, kernel_size=(1, 1),
                          stride=(1, 1), padding=(0, 0), groups=1, bias=bias)
        if self.bn_pw_out:
            self.pw2 = BatchNorm2d(num_features=out_channels, eps=bn_eps, momentum=bn_mom, track_running_stats=True)
        if self.act_pw_out:
            self.pw3 = PReLU(out_channels)

    def forward(self, x):
        x = self.dw1(x)
        if self.bn_dw_out:
            x = self.dw2(x)
        if self.act_dw_out:
            x = self.dw3(x)
        x = self.pw1(x)
        if self.bn_pw_out:
            x = self.pw2(x)
        if self.act_pw_out:
            x = self.pw3(x)
        return x


class VarGNet_Block(Module):
    def __init__(self, n_out_ch1, n_out_ch2, n_out_ch3, factor=2, dim_match=True, multiplier=1, kernel_size=(3, 3),
                 stride=(1, 1), dilation=1, with_dilate=False):
        super(VarGNet_Block, self).__init__()

        out_channels_1 = int(n_out_ch1 * multiplier)
        out_channels_2 = int(n_out_ch2 * multiplier)
        out_channels_3 = int(n_out_ch3 * multiplier)

        padding = (((kernel_size[0] - 1) * dilation + 1) // 2, ((kernel_size[1] - 1) * dilation + 1) // 2)

        if with_dilate:
            stride = (1, 1)
        self.dim_match = dim_match

        self.shortcut = Separable_Conv2d(in_channels=out_channels_1, out_channels=out_channels_3,
                                         kernel_size=kernel_size, padding=padding, stride=stride, factor=factor,
                                         bias=False, act_pw_out=False, dilation=dilation)
        self.sep1 = Separable_Conv2d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=kernel_size,
                                     padding=padding, stride=stride, factor=factor, bias=False, dilation=dilation)
        self.sep2 = Separable_Conv2d(in_channels=out_channels_2, out_channels=out_channels_3, kernel_size=kernel_size,
                                     padding=padding, stride=(1, 1), factor=factor, bias=False, act_pw_out=False,
                                     dilation=dilation)
        self.sep3 = Se_block(num_filter=out_channels_3)
        self.sep4 = PReLU(out_channels_3)

    def forward(self, x):
        if self.dim_match:
            short_cut = x
        else:
            short_cut = self.shortcut(x)
        x = self.sep1(x)
        x = self.sep2(x)
        if use_se:
            x = self.sep3(x)
        out = x + short_cut
        out = self.sep4(out)
        return out


class VarGNet_Branch_Merge_Block(Module):
    def __init__(self, n_out_ch1, n_out_ch2, n_out_ch3, factor=2, dim_match=False, multiplier=1, kernel_size=(3, 3),
                 stride=(2, 2), dilation=1, with_dilate=False):
        super(VarGNet_Branch_Merge_Block, self).__init__()

        out_channels_1 = int(n_out_ch1 * multiplier)
        out_channels_2 = int(n_out_ch2 * multiplier)
        out_channels_3 = int(n_out_ch3 * multiplier)

        padding = (((kernel_size[0] - 1) * dilation + 1) // 2, ((kernel_size[1] - 1) * dilation + 1) // 2)

        if with_dilate:
            stride = (1, 1)

        self.dim_match = dim_match

        self.shortcut = Separable_Conv2d(in_channels=out_channels_1, out_channels=out_channels_3,
                                         kernel_size=kernel_size, padding=padding, stride=stride, factor=factor,
                                         bias=False, act_pw_out=False, dilation=dilation)
        self.branch1 = Separable_Conv2d(in_channels=out_channels_1, out_channels=out_channels_2,
                                        kernel_size=kernel_size, padding=padding, stride=stride, factor=factor,
                                        bias=False, act_pw_out=False, dilation=dilation)
        self.branch2 = Separable_Conv2d(in_channels=out_channels_1, out_channels=out_channels_2,
                                        kernel_size=kernel_size, padding=padding, stride=stride, factor=factor,
                                        bias=False, act_pw_out=False, dilation=dilation)
        self.sep1 = PReLU(out_channels_2)
        self.sep2 = Separable_Conv2d(in_channels=out_channels_2, out_channels=out_channels_3, kernel_size=kernel_size,
                                     padding=padding, stride=(1, 1), factor=factor, bias=False, act_pw_out=False,
                                     dilation=dilation)
        self.sep3 = PReLU(out_channels_3)

    def forward(self, x):
        if self.dim_match:
            short_cut = x
        else:
            short_cut = self.shortcut(x)
        temp1 = self.branch1(x)
        temp2 = self.branch2(x)
        temp = temp1 + temp2
        temp = self.sep1(temp)
        temp = self.sep2(temp)
        out = temp + short_cut
        out = self.sep3(out)
        return out


class VarGNet_Conv_Block(Module):
    def __init__(self, stage, units, in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), multiplier=1,
                 factor=2, dilation=1, with_dilate=False):
        super(VarGNet_Conv_Block, self).__init__()

        assert stage >= 2, 'Stage is {}, stage must be set >=2'.format(stage)
        self.branch_merge = VarGNet_Branch_Merge_Block(n_out_ch1=in_channels, n_out_ch2=out_channels,
                                                       n_out_ch3=out_channels, factor=factor, dim_match=False,
                                                       multiplier=multiplier, kernel_size=kernel_size, stride=stride,
                                                       dilation=dilation, with_dilate=with_dilate)
        features = []
        for i in range(units - 1):
            features.append(
                VarGNet_Block(n_out_ch1=out_channels, n_out_ch2=out_channels, n_out_ch3=out_channels, factor=factor,
                              dim_match=True, multiplier=multiplier, kernel_size=kernel_size, stride=(1, 1),
                              dilation=dilation, with_dilate=with_dilate))
        self.features = Sequential(*features)

    def forward(self, x):
        x = self.branch_merge(x)
        x = self.features(x)
        return x


class Head_Block(Module):
    def __init__(self, num_filter, multiplier, head_pooling=False, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)):
        super(Head_Block, self).__init__()

        channels = int(num_filter * multiplier)

        self.head_pooling = head_pooling

        self.conv1 = Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=1, bias=False)
        # RGB图像包含3个通道（in_channels）
        self.bn1 = BatchNorm2d(num_features=channels, eps=bn_eps, momentum=bn_mom, track_running_stats=True)
        self.pool = PReLU(channels)
        self.head1 = MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.head2 = VarGNet_Block(n_out_ch1=num_filter, n_out_ch2=num_filter, n_out_ch3=num_filter, factor=1,
                                   dim_match=False, multiplier=multiplier, kernel_size=kernel_size, stride=(2, 2),
                                   dilation=1, with_dilate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        if self.head_pooling:
            x = self.head1(x)
        else:
            x = self.head2(x)
        return x


class Embedding_Block(Module):
    def __init__(self, input_channels, last_channels, emb_size, bias=False):
        super(Embedding_Block, self).__init__()

        self.input_channels = input_channels
        self.last_channels = last_channels

        # last channels(0, optional)
        self.conv0 = Conv2d(in_channels=input_channels, out_channels=last_channels, kernel_size=(1, 1), stride=(1, 1),
                            padding=(0, 0), bias=bias)
        self.bn0 = BatchNorm2d(num_features=last_channels, eps=bn_eps, momentum=bn_mom, track_running_stats=True)
        self.pool0 = PReLU(last_channels)

        # depthwise(1),输入为224*224时，可将kernel_size改为(14, 14)
        self.conv1 = Conv2d(in_channels=last_channels, out_channels=last_channels, kernel_size=(7, 7), stride=(1, 1),
                            padding=(0, 0), groups=int(last_channels / group_base), bias=bias)
        self.bn1 = BatchNorm2d(num_features=last_channels, eps=bn_eps, momentum=bn_mom, track_running_stats=True)

        # pointwise(2)
        self.conv2 = Conv2d(in_channels=last_channels, out_channels=last_channels // 2, kernel_size=(1, 1),
                            stride=(1, 1), padding=(0, 0), bias=bias)
        self.bn2 = BatchNorm2d(num_features=last_channels // 2, eps=bn_eps, momentum=bn_mom, track_running_stats=True)
        self.pool2 = PReLU(last_channels // 2)

        # FC
        self.fc = Linear(in_features=last_channels // 2, out_features=emb_size, bias=False)
        self.bn = BatchNorm1d(num_features=emb_size, eps=bn_eps, momentum=bn_mom, track_running_stats=True)

    def forward(self, x):
        if self.input_channels != self.last_channels:
            x = self.conv0(x)
            x = self.bn0(x)
            x = self.pool0(x)

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


class VarGFaceNet(Module):
    def __init__(self):
        super(VarGFaceNet, self).__init__()

        multiplier = 1.25
        emb_size = 512
        factor = 2
        head_pooling = False
        num_stage = 3
        stage_list = [2, 3, 4]
        units = [3, 7, 4]
        filter_list = [32, 64, 128, 256]
        last_channels = 1024
        dilation_list = [1, 1, 1]
        with_dilate_list = [False, False, False]

        self.head = Head_Block(num_filter=filter_list[0], multiplier=multiplier, head_pooling=head_pooling,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        body = []
        for i in range(num_stage):
            body.append(VarGNet_Conv_Block(stage=stage_list[i], units=units[i], in_channels=filter_list[i],
                                           out_channels=filter_list[i + 1], kernel_size=(3, 3), stride=(2, 2),
                                           multiplier=multiplier, factor=factor, dilation=dilation_list[i],
                                           with_dilate=with_dilate_list[i]))
        self.body = Sequential(*body)
        self.emb = Embedding_Block(input_channels=int(filter_list[3] * multiplier), last_channels=last_channels,
                                   emb_size=emb_size, bias=False)  # 源代码的input_channels缺少*multiplier，无法运行
        # initialization
        for m in self.modules():  # 借用MobileNetV3的初始化方法
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (BatchNorm1d, BatchNorm2d)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.emb(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)
        return output, norm
