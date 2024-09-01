import torch
from torch import nn
from torch.nn import init
import numpy as np


def get_timestep_embedding(timesteps, embedding_dim):
    timesteps = torch.flatten(timesteps)
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = torch.tensor([np.log(10000) / (half_dim - 1)])
    emb = torch.exp(torch.arange(half_dim).float() * -emb)
    emb = timesteps[:, None] * emb[None, :]  # outer product
    emb = torch.concat(tuple([torch.sin(emb), torch.cos(emb)]), dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.concat(emb, torch.zeros(timesteps.shape[0], 1), dim=1)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def determine_padding(input_img_size):
    if (input_img_size & (input_img_size - 1)) != 0:
        power_of_two = 1
        while power_of_two < input_img_size:
            power_of_two *= 2
        if input_img_size % 2 != 0:
            raise ValueError("no odd images allowed")
        x_pad = power_of_two - input_img_size
        y_pad = power_of_two - input_img_size
    else:
        x_pad = 0
        y_pad = 0
    return x_pad, y_pad


class DDPMConv3x3(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1, padding=None):
        super().__init__()
        if padding == None:
            padding = "same"
        self.conv = nn.Conv2d(
            channel_in, channel_out, 3, padding=padding, stride=stride
        )
        init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ResNet_Block(nn.Module):
    def __init__(self, act, ch_in, ch_out, ch_emb, skip_rescale, conv_shortcut) -> None:
        super().__init__()

        self.skip_rescale = skip_rescale
        self.conv_shortcut = conv_shortcut

        self.act = act
        self.conv0 = DDPMConv3x3(ch_in, ch_out)
        self.conv1 = DDPMConv3x3(ch_out, ch_out)
        self.ch_in, self.ch_out = ch_in, ch_out
        if ch_in != ch_out:
            self.skip_conv = DDPMConv3x3(ch_in, ch_out)  # used for mismatched_dim
        self.linear0 = nn.Linear(in_features=ch_emb, out_features=ch_out)
        self.linear1 = nn.Linear(in_features=ch_out, out_features=ch_out)
        self.groupnorm0 = nn.GroupNorm(
            num_channels=ch_in, num_groups=min(ch_in // 4, 32)
        )
        self.groupnorm1 = nn.GroupNorm(
            num_channels=ch_out, num_groups=min(ch_out // 4, 32)
        )
        self.dropout = nn.Dropout()

    def forward(self, x, temb):
        h = self.act(self.groupnorm0(x))
        h = self.conv0(h)

        if temb != None:
            h += self.linear0(self.act(temb)).unsqueeze(-1).unsqueeze(-1)

        h = self.act(self.groupnorm1(h))
        h = self.dropout(h)
        h = self.conv1(h)

        h += self.skip_conv(x) if self.ch_in != self.ch_out else x
        return h / np.sqrt(2.0) if self.skip_rescale else h


class Downsample(nn.Module):  # PP version uses a FIR filter...
    def __init__(self, ch_in, ch_out, with_conv: bool, input_img_size: int = None):
        super().__init__()
        self.with_conv = with_conv
        padding = (1, 1)
        self.down = (
            DDPMConv3x3(channel_in=ch_in, channel_out=ch_out, stride=2, padding=padding)
            if with_conv
            else nn.AvgPool2d(2, stride=2, padding=1)
        )

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):  # PP version uses a FIR filter...
    def __init__(self, ch_in, with_conv: bool):
        super().__init__()
        C = ch_in
        self.conv = DDPMConv3x3(C, C)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.up(x)
        if self.conv is not None:
            x = self.conv(x)
        return x
