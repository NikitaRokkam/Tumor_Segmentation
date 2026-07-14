"""
Attention U-Net with an ImageNet-pretrained EfficientNet-B4 encoder.

"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# Encoder stages we pull skip connections from, at strides /2, /4, /8, /16, /32
ENCODER_RETURN_NODES = {
    "features.1": "s2",
    "features.2": "s4",
    "features.3": "s8",
    "features.5": "s16",
    "features.7": "s32",  # bottleneck
}


class AttentionGate(nn.Module):
    def __init__(self, decoder_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_decoder = nn.Conv2d(decoder_channels, inter_channels, kernel_size=1)
        self.W_skip = nn.Conv2d(skip_channels, inter_channels, kernel_size=1)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        # decoder_feat and skip_feat are already the same spatial size by the time this is called
        combined = self.relu(self.W_decoder(decoder_feat) + self.W_skip(skip_feat))
        attention_map = self.psi(combined)          # [B, 1, H, W], values in (0, 1)
        return skip_feat * attention_map             # suppress irrelevant regions of the skip feature


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention_gate = AttentionGate(
            decoder_channels=out_channels, skip_channels=skip_channels, inter_channels=out_channels // 2
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Defensive guard for odd input sizes (EfficientNet stage sizes can be
        # off-by-one from exact powers of 2). At IMG_SIZE=256 this branch never
        # fires -- verified the shapes line up exactly -- so it's a no-op that
        # TorchScript tracing bakes out. If IMG_SIZE ever changes to a value
        # that isn't cleanly divisible by 32, re-trace and re-verify this path.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
        skip = self.attention_gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class AttentionUNet(nn.Module):
    def __init__(self, img_size: int = 256, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)
        self.encoder = create_feature_extractor(backbone, return_nodes=ENCODER_RETURN_NODES)

        # Infer channel counts at each stage with a dummy forward pass instead of hardcoding them.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feats = self.encoder(dummy)
        c2, c4, c8, c16, c32 = (feats[k].shape[1] for k in ["s2", "s4", "s8", "s16", "s32"])

        self.decoder4 = DecoderBlock(in_channels=c32, skip_channels=c16, out_channels=c16)
        self.decoder3 = DecoderBlock(in_channels=c16, skip_channels=c8, out_channels=c8)
        self.decoder2 = DecoderBlock(in_channels=c8, skip_channels=c4, out_channels=c4)
        self.decoder1 = DecoderBlock(in_channels=c4, skip_channels=c2, out_channels=c2)

        # Final upsample back to full input resolution (encoder stage s2 is at stride 2, not 1)
        self.final_upsample = nn.ConvTranspose2d(c2, c2 // 2, kernel_size=2, stride=2)
        self.head = nn.Conv2d(c2 // 2, 1, kernel_size=1)  # 1 output channel: raw logits for binary mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        d = self.decoder4(feats["s32"], feats["s16"])
        d = self.decoder3(d, feats["s8"])
        d = self.decoder2(d, feats["s4"])
        d = self.decoder1(d, feats["s2"])
        d = self.final_upsample(d)
        return self.head(d)  # logits, shape [B, 1, H, W] -- sigmoid applied in the loss/inference, not here
