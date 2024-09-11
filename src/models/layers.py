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


class Conv3x3(nn.Module):
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
    def __init__(
        self,
        shape,
        act,
        ch_in,
        ch_out,
        ch_emb,
        skip_rescale,
        dropout,
        conv_shortcut=False,
        num_heads=0,
    ) -> None:
        super().__init__()

        self.skip_rescale = skip_rescale
        self.conv_shortcut = conv_shortcut
        self.num_heads = num_heads
        self.attn_emb_dim = shape[2] * shape[3]

        self.act = act
        self.conv0 = Conv3x3(ch_in, ch_out)
        if conv_shortcut:
            self.conv1 = Conv3x3(ch_out, ch_out)
        else:
            print(shape[1] * shape[2] * shape[3])
            self.linear1 = nn.Linear(
                in_features=shape[1] * shape[2] * shape[3],
                out_features=shape[1] * shape[2] * shape[3],
            )
        self.ch_in, self.ch_out = ch_in, ch_out
        if ch_in != ch_out:
            self.skip_conv = Conv3x3(ch_in, ch_out)  # used for mismatched_dim
        self.linear0 = nn.Linear(in_features=ch_emb, out_features=ch_out)

        self.groupnorm0 = nn.GroupNorm(
            num_channels=ch_in, num_groups=min(ch_in // 4, 32)
        )
        self.groupnorm1 = nn.GroupNorm(
            num_channels=ch_out, num_groups=min(ch_out // 4, 32)
        )
        self.dropout = nn.Dropout(dropout)

        if self.num_heads:
            self.norm2 = nn.GroupNorm(
                num_channels=ch_out, num_groups=min(ch_out // 4, 32)
            )
            self.qkv = Conv3x3(ch_out, ch_out * 3)
            self.MHA = nn.MultiheadAttention(
                embed_dim=self.attn_emb_dim, num_heads=num_heads
            )

    def forward(self, x, temb):
        h = self.act(self.groupnorm0(x))
        h = self.conv0(h)

        if temb != None:
            h += self.linear0(self.act(temb)).unsqueeze(-1).unsqueeze(-1)

        h = self.act(self.groupnorm1(h))
        h = self.dropout(h)

        if self.conv_shortcut:
            h = self.conv1(h)
        else:
            h_shape = h.shape
            h = self.linear1(h.reshape(h_shape[0], -1)).reshape(*h_shape)

        x = h + self.skip_conv(x) if self.ch_in != self.ch_out else x
        if self.skip_rescale:
            x = x / np.sqrt(2.0)

        with torch.autograd.set_detect_anomaly(True):
            if self.num_heads:
                x = self.norm2(x)
                Q, K, V = torch.reshape(
                    self.qkv(x), (x.shape[0], x.shape[1], x.shape[2] * x.shape[3], 3)
                ).unbind(3)
                img_dim = int(np.sqrt(self.attn_emb_dim))
                x = x + self.MHA(Q, K, V, need_weights=False)[0].reshape(
                    x.shape[0], x.shape[1], img_dim, img_dim
                )
                if self.skip_rescale:
                    x = x / np.sqrt(2.0)
        return x


class Downsample(nn.Module):  # PP version uses a FIR filter...
    def __init__(self, ch_in, ch_out, with_conv: bool, input_img_size: int = None):
        super().__init__()
        self.with_conv = with_conv
        padding = (1, 1)
        self.down = (
            Conv3x3(channel_in=ch_in, channel_out=ch_out, stride=2, padding=padding)
            if with_conv
            else nn.AvgPool2d(2, stride=2, padding=1)
        )

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):  # PP version uses a FIR filter...
    def __init__(self, ch_in, with_conv: bool):
        super().__init__()
        C = ch_in
        self.up_conv = nn.ConvTranspose2d(C, C, 4, stride=2, padding=1)

    def forward(self, x):
        return self.up_conv(x)
