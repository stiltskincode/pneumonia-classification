import torch
import torch.nn as nn
from torchvision import models


# SE Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        se = self.avg_pool(x).view(batch_size, channels)
        se = self.fc(se).view(batch_size, channels, 1, 1)
        return x * se


# CBAM
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(in_channels, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.channel_attention(x)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_attention = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        return x * spatial_attention


class UNetWithResNetAttentionCBAM(nn.Module):
    def __init__(self, resnet_type="resnet152", num_classes=2, dropout_rate=0.1):
        super(UNetWithResNetAttentionCBAM, self).__init__()

        # Load ResNet Encoder
        if resnet_type == "resnet18":
            resnet = models.resnet18(pretrained=True)
            encoder_channels = [64, 64, 128, 256, 512]
        elif resnet_type == "resnet34":
            resnet = models.resnet34(pretrained=True)
            encoder_channels = [64, 64, 128, 256, 512]
        elif resnet_type == "resnet50":
            resnet = models.resnet50(pretrained=True)
            encoder_channels = [64, 256, 512, 1024, 2048]
        elif resnet_type == "resnet101":
            resnet = models.resnet101(pretrained=True)
            encoder_channels = [64, 256, 512, 1024, 2048]
        elif resnet_type == "resnet152":
            resnet = models.resnet152(pretrained=True)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # Modify first convolution for 2-channel input
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Encoder with CBAM
        self.encoder = nn.ModuleDict({
            "enc1": nn.Sequential(*list(resnet.children())[:4], CBAM(64)),
            "enc2": nn.Sequential(resnet.layer1, CBAM(encoder_channels[1])),
            "enc3": nn.Sequential(resnet.layer2, CBAM(encoder_channels[2])),
            "enc4": nn.Sequential(resnet.layer3, CBAM(encoder_channels[3])),
            "enc5": nn.Sequential(resnet.layer4, CBAM(encoder_channels[4])),
        })

        # Bottleneck with CBAM
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(1024),
        )

        # Decoder with Attention Blocks
        self.decoder = nn.ModuleDict({
            "upconv4": nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            "dec4": self._decoder_block(encoder_channels[-2] + 512, 512, dropout_rate, attention=True),
            "upconv3": nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            "dec3": self._decoder_block(encoder_channels[-3] + 256, 256, dropout_rate, attention=True),
            "upconv2": nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            "dec2": self._decoder_block(encoder_channels[-4] + 128, 128, dropout_rate, attention=True),
            "upconv1": nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            "dec1": self._decoder_block(encoder_channels[-5] + 64, 64, dropout_rate, attention=True),
            "final_conv": nn.Conv2d(64, num_classes, kernel_size=1),
        })

    def _decoder_block(self, in_channels, out_channels, dropout_rate=0.1, attention=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if attention:
            layers.append(CBAM(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, mask):
        # Combine image and bounding box mask as input
        x = torch.cat([x, mask], dim=1)

        # Encoder
        enc1 = self.encoder["enc1"](x)
        enc2 = self.encoder["enc2"](enc1)
        enc3 = self.encoder["enc3"](enc2)
        enc4 = self.encoder["enc4"](enc3)
        enc5 = self.encoder["enc5"](enc4)

        # Bottleneck
        bottleneck = self.bottleneck(enc5)

        # Decoder
        dec4 = self.decoder["upconv4"](bottleneck)
        dec4 = self.decoder["dec4"](torch.cat([dec4, enc4], dim=1))
        dec3 = self.decoder["upconv3"](dec4)
        dec3 = self.decoder["dec3"](torch.cat([dec3, enc3], dim=1))
        dec2 = self.decoder["upconv2"](dec3)
        dec2 = self.decoder["dec2"](torch.cat([dec2, enc2], dim=1))
        dec1 = self.decoder["upconv1"](dec2)
        enc1_resized = torch.nn.functional.interpolate(enc1, size=dec1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.decoder["dec1"](torch.cat([dec1, enc1_resized], dim=1))

        # Final output with 2 channels (pneumonia mask + bounding box mask)
        return self.decoder["final_conv"](dec1)

