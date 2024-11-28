import torch
import torch.nn as nn
from torchvision import models


class UNetWithResNet(nn.Module):
    def __init__(self, resnet_type="resnet152", num_classes=1):
        super(UNetWithResNet, self).__init__()

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

        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.encoder = nn.ModuleDict({
            "enc1": nn.Sequential(*list(resnet.children())[:4]),
            "enc2": resnet.layer1,
            "enc3": resnet.layer2,
            "enc4": resnet.layer3,
            "enc5": resnet.layer4
        })

        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 1024, kernel_size=3, padding=1),  # Zmniejszenie z max kanałów ResNet
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.ModuleDict({
            "upconv4": nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            "dec4": self._decoder_block(encoder_channels[-2] + 512, 512),
            "upconv3": nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            "dec3": self._decoder_block(encoder_channels[-3] + 256, 256),
            "upconv2": nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            "dec2": self._decoder_block(encoder_channels[-4] + 128, 128),
            "upconv1": nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            "dec1": self._decoder_block(encoder_channels[-5] + 64, 64),
            "final_conv": nn.Conv2d(64, num_classes, kernel_size=1)
        })

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
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

        return self.decoder["final_conv"](dec1)

