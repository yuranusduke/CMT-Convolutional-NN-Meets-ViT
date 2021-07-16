"""
This file contains all operations about building CMT model
paper: <CMT: Convolutional Neural Networks Meet Vision Transformers>
addr: https://arxiv.org/abs/2107.06263

NOTE: In the paper, in Table 1, authors denote Patch Aggregation and LPU may have different channels,
but in practice, in LPU, residual connection is need, so difference in channels may make summation
impossible, therefore, we use the same channels between Patch Aggregation and LPU. So calculated number of
params differ a little from the paper.

Update: The new version of architecture is released to fix the above bugs.

Created by Kunhong Yu
Date: 2021/07/14
"""

import torch as t
from torch.nn import functional as F
from model.modules import Stem, PatchAggregation, CMTBlock


#########################
#   CMT Configuration   #
#########################
class CMT(t.nn.Module):
    """Define CMT model"""

    def __init__(self,
                 in_channels = 3,
                 stem_channels = 16,
                 cmt_channelses = [46, 92, 184, 368],
                 pa_channelses = [46, 92, 184, 368],
                 R = 3.6,
                 repeats = [2, 2, 10, 2],
                 input_size = 224,
                 num_classes = 1000):
        """
        Args :
            --in_channels: default is 3
            --stem_channels: stem channels, default is 16
            --cmt_channelses: list, default is [46, 92, 184, 368]
            --pa_channels: patch aggregation channels, list, default is [46, 92, 184, 368]
            --R: expand ratio, default is 3.6
            --repeats: list, to specify how many CMT blocks stacked together, default is [2, 2, 10, 2]
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT, self).__init__()

        if input_size == 224:
            sizes = [56, 28, 14, 7]
        elif input_size == 160:
            sizes = [40, 20, 10, 5]
        elif input_size == 192:
            sizes = [48, 24, 12, 6]
        elif input_size == 256:
            sizes = [64, 32, 16, 8]
        elif input_size == 288:
            sizes = [72, 36, 18, 9]
        else:
            raise Exception('No other input sizes!')

        # 1. Stem
        self.stem = Stem(in_channels = in_channels, out_channels = stem_channels)

        # 2. Patch Aggregation 1
        self.pa1 = PatchAggregation(in_channels = stem_channels, out_channels = pa_channelses[0])
        self.pa2 = PatchAggregation(in_channels = cmt_channelses[0], out_channels = pa_channelses[1])
        self.pa3 = PatchAggregation(in_channels = cmt_channelses[1], out_channels = pa_channelses[2])
        self.pa4 = PatchAggregation(in_channels = cmt_channelses[2], out_channels = pa_channelses[3])

        # 3. CMT block
        cmt1 = []
        for _ in range(repeats[0]):
            cmt_layer = CMTBlock(input_size = sizes[0],
                                 kernel_size = 8,
                                 d_k = cmt_channelses[0],
                                 d_v = cmt_channelses[0],
                                 num_heads = 1,
                                 R = R, in_channels = pa_channelses[0])
            cmt1.append(cmt_layer)
        self.cmt1 = t.nn.Sequential(*cmt1)

        cmt2 = []
        for _ in range(repeats[1]):
            cmt_layer = CMTBlock(input_size = sizes[1],
                                 kernel_size = 4,
                                 d_k = cmt_channelses[1] // 2,
                                 d_v = cmt_channelses[1] // 2,
                                 num_heads = 2,
                                 R = R, in_channels = pa_channelses[1])
            cmt2.append(cmt_layer)
        self.cmt2 = t.nn.Sequential(*cmt2)

        cmt3 = []
        for _ in range(repeats[2]):
            cmt_layer = CMTBlock(input_size = sizes[2],
                                 kernel_size = 2,
                                 d_k = cmt_channelses[2] // 4,
                                 d_v = cmt_channelses[2] // 4,
                                 num_heads = 4,
                                 R = R, in_channels = pa_channelses[2])
            cmt3.append(cmt_layer)
        self.cmt3 = t.nn.Sequential(*cmt3)

        cmt4 = []
        for _ in range(repeats[3]):
            cmt_layer = CMTBlock(input_size = sizes[3],
                                 kernel_size = 1,
                                 d_k = cmt_channelses[3] // 8,
                                 d_v = cmt_channelses[3] // 8,
                                 num_heads = 8,
                                 R = R, in_channels = pa_channelses[3])
            cmt4.append(cmt_layer)
        self.cmt4 = t.nn.Sequential(*cmt4)

        # 4. Global Avg Pool
        self.avg = t.nn.AdaptiveAvgPool2d(1)

        # 5. FC
        self.fc = t.nn.Sequential(
            t.nn.Linear(cmt_channelses[-1], 1280),
            t.nn.ReLU(inplace = True) # we use ReLU here as default
        )

        # 6. Classifier
        self.classifier = t.nn.Sequential(
            t.nn.Linear(1280, num_classes)
        )

    def forward(self, x):

        # 1. Stem
        x_stem = self.stem(x)

        # 2. PA1 + CMTb1
        x_pa1 = self.pa1(x_stem)
        x_cmtb1 = self.cmt1(x_pa1)

        # 3. PA2 + CMTb2
        x_pa2 = self.pa2(x_cmtb1)
        x_cmtb2 = self.cmt2(x_pa2)

        # 4. PA3 + CMTb3
        x_pa3 = self.pa3(x_cmtb2)
        x_cmtb3 = self.cmt3(x_pa3)

        # 5. PA4 + CMTb4
        x_pa4 = self.pa4(x_cmtb3)
        x_cmtb4 = self.cmt4(x_pa4)

        # 6. Avg
        x_avg = self.avg(x_cmtb4)
        x_avg = x_avg.squeeze()

        # 7. Linear + Classifier
        x_fc = self.fc(x_avg)
        out = self.classifier(x_fc)

        return out



#########################
#      CMT Models       #
#########################
# 1. CMT-Ti
class CMT_Ti(t.nn.Module):
    """Define CMT-Ti model"""

    def __init__(self, in_channels = 3, input_size = 224, num_classes = 1000):
        """
        Args :
            --in_channels: default is 3
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT_Ti, self).__init__()

        self.cmt_ti = CMT(in_channels = in_channels,
                          stem_channels = 16,
                          cmt_channelses = [46, 92, 184, 368],
                          pa_channelses = [46, 92, 184, 368],
                          R = 3.6,
                          repeats = [2, 2, 10, 2],
                          input_size = input_size,
                          num_classes = num_classes)

    def forward(self, x):

        x = self.cmt_ti(x)

        return x


# 2. CMT-XS
class CMT_XS(t.nn.Module):
    """Define CMT-XS model"""

    def __init__(self, in_channels = 3, input_size = 224, num_classes = 1000):
        """
        Args :
            --in_channels: default is 3
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT_XS, self).__init__()

        self.cmt_xs = CMT(in_channels = in_channels,
                          stem_channels = 16,
                          cmt_channelses = [52, 104, 208, 416],
                          pa_channelses = [52, 104, 208, 416],
                          R = 3.8,
                          repeats = [3, 3, 12, 3],
                          input_size = input_size,
                          num_classes = num_classes)

    def forward(self, x):

        x = self.cmt_xs(x)

        return x


# 3. CMT-S
class CMT_S(t.nn.Module):
    """Define CMT-S model"""

    def __init__(self, in_channels = 3, input_size = 224, num_classes = 1000):
        """
        Args :
            --in_channels: default is 3
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT_S, self).__init__()

        self.cmt_s = CMT(in_channels = in_channels,
                         stem_channels = 32,
                         cmt_channelses = [64, 128, 256, 512],
                         pa_channelses = [64, 128, 256, 512],
                         R = 4.,
                         repeats = [3, 3, 16, 3],
                         input_size = input_size,
                         num_classes = num_classes)

    def forward(self, x):

        x = self.cmt_s(x)

        return x


# 4. CMT-B
class CMT_B(t.nn.Module):
    """Define CMT-B model"""

    def __init__(self, in_channels = 3, input_size = 224, num_classes = 1000):
        """
        Args :
            --in_channels: default is 3
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CMT_B, self).__init__()

        self.cmt_b = CMT(in_channels = in_channels,
                         stem_channels = 38,
                         cmt_channelses = [76, 152, 304, 608],
                         pa_channelses = [76, 152, 304, 608],
                         R = 4.,
                         repeats = [4, 4, 20, 4],
                         input_size = input_size,
                         num_classes = num_classes)

    def forward(self, x):

        x = self.cmt_b(x)

        return x
