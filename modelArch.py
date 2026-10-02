import torch
import torch.nn as nn


def get_time_embedding(time_steps, temb_dim):
    factor = 10000 ** (
        (torch.arange(start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.t_emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t_emb):
        out = self.conv1(x)
        out = out + self.t_emb_layer(t_emb)[:, :, None, None]
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim):
        super().__init__()
        self.res_block = ResnetBlock(in_channels, out_channels, t_emb_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        feat = self.res_block(x, t_emb)
        out = self.downsample(feat)
        return out, feat


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.res_block = ResnetBlock(in_channels * 2, out_channels, t_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        out = self.res_block(x, t_emb)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels=1, t_emb_dim=128):
        super().__init__()
        self.t_emb_dim = t_emb_dim

        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )

        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.enc1 = EncoderBlock(64, 128, t_emb_dim)
        self.enc2 = EncoderBlock(128, 256, t_emb_dim)

        self.bottleneck1 = ResnetBlock(256, 256, t_emb_dim)
        self.bottleneck2 = ResnetBlock(256, 256, t_emb_dim)
        self.bottleneck3 = ResnetBlock(256, 256, t_emb_dim)

        self.dec1 = DecoderBlock(256, 128, t_emb_dim)
        self.dec2 = DecoderBlock(128, 64, t_emb_dim)

        self.out_block = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        x_in = self.conv_in(x)

        x1, skip1 = self.enc1(x_in, t_emb)
        x2, skip2 = self.enc2(x1, t_emb)

        b = self.bottleneck1(x2, t_emb)
        b = self.bottleneck2(b, t_emb)
        b = self.bottleneck3(b, t_emb)

        d1 = self.dec1(b, skip2, t_emb)
        d2 = self.dec2(d1, skip1, t_emb)

        noise_pred = self.out_block(d2)
        return noise_pred