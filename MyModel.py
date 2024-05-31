import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        '''
        The code will be made public after the paper is accepted.
        '''

class PMSFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=True, BatchNorm=nn.BatchNorm2d):
        super(PMSFM, self).__init__()
        '''
        The code will be made public after the paper is accepted.
        '''

    def forward(self, x, y):
        '''
        The code will be made public after the paper is accepted.
        '''


class Unsample(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        '''
        The code will be made public after the paper is accepted.
        '''

    def forward(self, x):
        '''
        The code will be made public after the paper is accepted.
        '''

class Sample(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        '''
        The code will be made public after the paper is accepted.
        '''

    def forward(self, x):
        '''
        The code will be made public after the paper is accepted.
        '''

class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        '''
        The code will be made public after the paper is accepted.
        '''

    def forward(self, x):
        '''
        The code will be made public after the paper is accepted.
        '''

class AP(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        '''
        The code will be made public after the paper is accepted.
        '''

    def stoken_forward(self, x):
        '''
        The code will be made public after the paper is accepted.
        '''

    def direct_forward(self, x):
        '''
        The code will be made public after the paper is accepted.
        '''

    def forward(self, x):
        '''
        The code will be made public after the paper is accepted.
        '''

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        '''
        The code will be made public after the paper is accepted.
        '''

    def forward(self, x):
        '''
        The code will be made public after the paper is accepted.
        '''


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        '''
        The code will be made public after the paper is accepted.
        '''

    def forward(self, x):
        '''
        The code will be made public after the paper is accepted.
        '''


class SEWM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(SEWM, self).__init__()
        '''
        The code will be made public after the paper is accepted.
        '''

    def forward(self, x):
        '''
        The code will be made public after the paper is accepted.
        '''

class APOENet(nn.Module):
    def __init__(self):
        super(APOENet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, apoe):
        apoe = apoe.squeeze(1)
        return self.fc(apoe)

# sMRINet class (input: torch.Size([32, 166, 256, 256])NCHW, output: 3 classes)
class sMRINet(nn.Module):
    def __init__(self):
        super(sMRINet, self).__init__()
        # input: torch.Size([32, 166, 256, 256])NCHW
        '''
        The code will be made public after the paper is accepted.
        '''

    def forward(self, npy, apoe):
        '''
        The code will be made public after the paper is accepted.
        '''
    
if __name__ == '__main__':
    sMRINet = sMRINet()
    print('sMRINet:', sMRINet)
    for name, parameter in sMRINet.named_parameters():
        print(f"{name}: {parameter.numel()} parameters")
    # all parameter num
    print('all parameter num:', sum(p.numel() for p in sMRINet.parameters()))
    print('sMRINet(torch.randn(32, 166, 256, 256)):', sMRINet(torch.randn(32, 166, 256, 256), torch.nn.functional.one_hot(torch.randint(0, 6, (32, 1)), num_classes=6).float()))