import torch
from torch import nn
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class ConvDecoderBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        emb_dim=64,
    ):
        super().__init__()

        self.upsample = ConvDecoderBlock(emb_dim, emb_dim, channel, n_res_block, n_res_channel, stride=4)
        self.dec = ConvDecoderBlock(
            emb_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        '''
            input: (b, t, num_patches, emb_dim)
        '''
        num_patches = input.shape[2]
        num_side_patches = int(num_patches ** 0.5)
        input = rearrange(input, "b t (h w) e -> (b t) h w e", h=num_side_patches, w=num_side_patches)
        quant = input.permute(0, 3, 1, 2)
        dec = self.decode(quant)
        return dec

    def decode(self, quant):
        upsampled = self.upsample(quant)
        dec = self.dec(upsampled)
        return dec
