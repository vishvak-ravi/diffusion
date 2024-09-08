import torch
from torch import nn
from numpy import arange
from .layers import (
    DDPMConv3x3,
    ResNet_Block,
    Downsample,
    Upsample,
    get_timestep_embedding,
    determine_padding,
)


class DiffusionNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.data_centered = config.data_centered
        self.channel_multipliers = config.channel_multipliers
        self.num_res_blocks = config.num_res_blocks
        self.attn_resolutions = config.attn_resolutions
        self.nf = config.nf  # num features
        self.num_attn_heads = config.num_attn_heads
        self.skip_rescale = config.skip_rescale
        self.conv_shortcut = config.conv_shortcut
        self.down_with_conv = config.down_with_conv
        self.img_channels = config.img_channels
        self.img_dims = config.img_dims
        self.dropout = config.dropout

        if config.act == "silu":
            self.act = nn.SiLU()
        self.channel_sizes = [self.nf]
        for i in range(len(self.channel_multipliers)):
            self.channel_sizes.append(
                self.channel_sizes[i] * self.channel_multipliers[i]
            )
        self.num_resolutions = len(self.channel_sizes)

        self.t_emb_linear = nn.Sequential(
            nn.Linear(self.nf, 4 * self.nf),
            self.act,
            nn.Linear(4 * self.nf, 4 * self.nf),
        )

        # create_blocks
        power_of_two_img_size = config.img_size + determine_padding(config.img_size)[0]
        self.downsampler = nn.ModuleDict(
            {
                "initial_conv": nn.ModuleList(
                    [
                        DDPMConv3x3(
                            self.img_channels,
                            self.nf,
                        ),
                    ]
                ),
                "resnetblocks": nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                ResNet_Block(
                                    (
                                        config.batch_size,
                                        self.channel_sizes[i],
                                        self.img_dims[i + 1],
                                        self.img_dims[i + 1],
                                    ),
                                    self.act,
                                    self.channel_sizes[i],
                                    self.channel_sizes[
                                        i + (block_idx == self.num_res_blocks - 1)
                                    ],
                                    self.nf * 4,
                                    self.skip_rescale,
                                    conv_shortcut=True,
                                    num_heads=(
                                        self.num_attn_heads
                                        if self.img_dims[i + 1] in self.attn_resolutions
                                        else 0
                                    ),
                                    dropout=self.dropout,
                                )
                                for block_idx in range(self.num_res_blocks)
                            ]
                        )
                        for i in range(self.num_resolutions - 1)
                    ]
                ),
                "downsamplers": nn.ModuleList(
                    [
                        Downsample(
                            ch_in=self.channel_sizes[i],
                            ch_out=self.channel_sizes[i],
                            with_conv=self.down_with_conv,
                        )
                        for i in range(1, len(self.channel_sizes))
                    ]
                ),
            }
        )

        self.bridge = nn.ModuleDict(
            {
                "resnet_1": ResNet_Block(
                    (
                        config.batch_size,
                        self.channel_sizes[-1],
                        self.img_dims[-1],
                        self.img_dims[-1],
                    ),
                    self.act,
                    ch_in=self.channel_sizes[-1],
                    ch_out=self.channel_sizes[-1],
                    ch_emb=self.nf * 4,
                    skip_rescale=self.skip_rescale,
                    num_heads=self.num_attn_heads,
                    conv_shortcut=False,
                    dropout=self.dropout,
                ),
                "resnet_2": ResNet_Block(
                    (
                        config.batch_size,
                        self.channel_sizes[-1],
                        self.img_dims[-1],
                        self.img_dims[-1],
                    ),
                    self.act,
                    ch_in=self.channel_sizes[-1],
                    ch_out=self.channel_sizes[-1],
                    ch_emb=self.nf * 4,
                    skip_rescale=self.skip_rescale,
                    num_heads=self.num_attn_heads,
                    conv_shortcut=False,
                    dropout=self.dropout,
                ),
            }
        )

        self.upsampler = nn.ModuleDict(
            {
                "final_conv": DDPMConv3x3(self.nf, 3),
                "resnetblocks": nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                ResNet_Block(
                                    (
                                        config.batch_size,
                                        2 * self.channel_sizes[i + (block_idx == 0)],
                                        self.img_dims[i + 1],
                                        self.img_dims[i + 1],
                                    ),
                                    self.act,
                                    2 * self.channel_sizes[i + (block_idx == 0)],
                                    self.channel_sizes[i],
                                    self.nf * 4,
                                    self.skip_rescale,
                                    conv_shortcut=True,
                                    num_heads=(
                                        self.num_attn_heads
                                        if self.img_dims[i] in self.attn_resolutions
                                        else 0
                                    ),
                                    dropout=self.dropout,
                                )
                                for block_idx in range(self.num_res_blocks + 1)
                            ]
                        )
                        for i in reversed(range(self.num_resolutions - 1))
                    ]
                ),
                "upsamplers": nn.ModuleList(
                    [
                        Upsample(ch_in=channel_size, with_conv=False)
                        for channel_size in self.channel_sizes[::-1][1:]
                    ]
                ),
            }
        )

        self.final_conv = DDPMConv3x3(self.nf, self.img_channels)

    def forward(self, x, timesteps, labels=None):

        # config
        if self.data_centered:
            h = x
        else:
            h = x - 0.5

        # time steps
        time_embeddings = get_timestep_embedding(timesteps, self.nf).float()
        t_emb = self.t_emb_linear(time_embeddings)

        # downsample
        for module in self.downsampler["initial_conv"]:
            h = module(h)
        hs = [h]
        for i_level in range(self.num_resolutions - 1):
            for i_block in range(self.num_res_blocks):
                h = self.downsampler["resnetblocks"][i_level][i_block](hs[-1], t_emb)
                hs.append(h)
            if i_level != self.num_resolutions - 2:
                hs.append(self.downsampler["downsamplers"][i_level](hs[-1]))

        h = hs[-1]
        h = self.bridge["resnet_1"](h, t_emb)
        h = self.bridge["resnet_2"](h, t_emb)

        # upsample
        for i_level in range(self.num_resolutions - 1):
            for i_block in range(self.num_res_blocks + 1):
                h = torch.concat(tuple([h, hs.pop()]), dim=1)
                h = self.upsampler["resnetblocks"][i_level][i_block](h, t_emb)
            if i_level != self.num_resolutions - 2:
                h = self.upsampler["upsamplers"][i_level](h)

        h = self.final_conv(h)

        return h
