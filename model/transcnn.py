"""
This file contains all operations about building CMT model
paper: <Transformer in Convolutional Neural Networks>
addr: https://arxiv.org/abs/2106.03180

Created by Kunhong Yu
Date: 2021/07/22
"""

import torch as t
from torch.nn import functional as F
from model.modules import ConvBlock, TransCNNBlock, TDB


#########################
#   TransCNN Building   #
#########################
class TransCNN(t.nn.Module):
    """Define TransCNN model"""

    def __init__(self,
                 num_classes,
                 in_channels = 3,
                 g_sizes = [[8, 4, 2], [7, 4, 2], [3, 2, 2], [2, 2, 2]],
                 exp_ratios = [4, 4, 6, 5],
                 repeats = [2, 2, 2, 2]):
        """
        Args :
            --num_classes
            --in_channels: default is 3
            --g_sizes: default is [[8, 4, 2], [7, 4, 2], [3, 2, 2], [2, 2, 2]] according to part 4.1 in paper
            --exp_ratios: expansion ratios, default is [4, 4, 6, 5]
            --repeats: list, to specify how many CMT blocks stacked together, default is [2, 2, 2, 2]
        """
        super(TransCNN, self).__init__()

        # 1. ConvBlock
        self.conv = ConvBlock(in_channels = in_channels)
        in_channels = 64
        embeds = [64, 128, 256, 512]

        # 2. Build modules
        stages = []
        for l in range(4): # 4 stages
            repeat = repeats[l]
            embed = embeds[l]
            stage = []
            for _ in range(repeat):
                layer = TransCNNBlock(in_channels = in_channels,
                                      g_size = g_sizes[l],
                                      exp_ratio = exp_ratios[l],
                                      kernel_size = 3 if l % 2 else 5,
                                      embed_dim = embed)

                stage.append(layer)



            stage.append(TDB(in_channels, embed))
            in_channels = embed

            stage = t.nn.Sequential(*stage)
            stages.append(stage)

        self.stages = t.nn.ModuleList(stages)

        # 3. Global Avg Pool
        self.avg = t.nn.AdaptiveAvgPool2d(1)

        # 4. Classifier
        self.cls = t.nn.Sequential(
            t.nn.Linear(512, num_classes),
        )

    def forward(self, x):

        # 1. Conv
        x_conv = self.conv(x)
        x = x_conv

        # 2. TransCNN Blocks
        for l in range(4):
            transcnn_block = self.stages[l]
            x = transcnn_block(x)

        # 3. Avg
        x_avg = self.avg(x)
        x_avg = x_avg.squeeze()

        # 7. Linear + Classifier
        out = self.cls(x_avg)

        return out


#########################
#   TransCNN Models     #
#########################
def TransCNN_Small(num_classes, in_channels, g_sizes = [[4, 4, 2], [4, 4, 2], [4, 2, 2], [2, 2, 2]]):
    """TransCNN Small model
    Args :
        --num_classes
        --in_channels
        --g_sizes: default is [[8, 4, 2], [7, 4, 2], [3, 2, 2], [2, 2, 2]]
    """
    model = TransCNN(num_classes,
                     in_channels = in_channels,
                     exp_ratios = [4, 4, 6, 5],
                     repeats = [2, 2, 2, 2],
                     g_sizes = g_sizes)

    return model

def TransCNN_Base(num_classes, in_channels, g_sizes = [[4, 4, 2], [4, 4, 2], [4, 2, 2], [2, 2, 2]]):
    """TransCNN Base model
    Args :
        --num_classes
        --in_channel
        --g_sizes: default is [[8, 4, 2], [7, 4, 2], [3, 2, 2], [2, 2, 2]]
    """
    model = TransCNN(num_classes,
                     in_channels = in_channels,
                     exp_ratios = [4, 4, 6, 6],
                     repeats = [3, 4, 8, 3],
                     g_sizes = g_sizes)

    return model

def TransCNN_Model(num_classes, in_channels, g_sizes = [[4, 4, 2], [4, 4, 2], [4, 2, 2], [2, 2, 2]], type = 'base'):
    """Get TransCNN model
    Args :
        --num_classes
        --in_channels
        --g_sizes: default is [[8, 4, 2], [7, 4, 2], [3, 2, 2], [2, 2, 2]]
        --type: 'base' or 'small'
    """
    assert type in ['small', 'base']

    if type == 'base': # base
        model = TransCNN_Base(num_classes = num_classes, in_channels = in_channels, g_sizes = g_sizes)
    else: # small
        model = TransCNN_Small(num_classes = num_classes, in_channels = in_channels, g_sizes = g_sizes)

    return model



## Unit test
if __name__ == '__main__':
    num_classes = 1000
    in_channels = 3
    input_size = 256 # I can only get this results from paper

    small_model = TransCNN_Model(num_classes, in_channels = in_channels, type = 'small').cuda()
    print('TransCNN-Small : \n', small_model)
    base_model = TransCNN_Model(num_classes, in_channels = in_channels, type = 'base').cuda()
    print('TransCNN-Base : \n', base_model)

    total = sum(p.numel() for p in small_model.parameters())
    print("Total params for TransCNN-Small: %.2fM" % (total / 1e6))
    total = sum(p.numel() for p in base_model.parameters())
    print("Total params for TransCNN-Base: %.2fM" % (total / 1e6))

    rand_input = t.randn(2, 3, input_size, input_size).cuda()
    small_out = small_model(rand_input)
    base_out = base_model(rand_input)