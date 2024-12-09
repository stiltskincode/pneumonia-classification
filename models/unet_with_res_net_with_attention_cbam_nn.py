import torch
import torch.nn as nn
from torchvision import models


# Mechanizm uwagi SEBlock
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.pool(x).view(batch, channels)
        y = self.fc1(y)
        y = nn.ReLU()(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y


class UNetWithResNet(nn.Module):
    def __init__(self, resnet_type="resnet152", num_classes=1, dropout_rate=0.1, bounding_box_dropout=0.5, img_size=512):
        super(UNetWithResNet, self).__init__()

        self.bounding_box_dropout = bounding_box_dropout

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

        # Zamiana pierwszej warstwy na dane z 2 kanałami
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Encoder z mechanizmem uwagi SEBlock
        self.encoder = nn.ModuleDict({
            "enc1": nn.Sequential(*list(resnet.children())[:4], SEBlock(64)),
            "enc2": nn.Sequential(resnet.layer1, SEBlock(256)),
            "enc3": nn.Sequential(resnet.layer2, SEBlock(512)),
            "enc4": nn.Sequential(resnet.layer3, SEBlock(1024)),
            "enc5": nn.Sequential(resnet.layer4, SEBlock(2048))
        })

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.ModuleDict({
            "upconv4": nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            "dec4": self._decoder_block(encoder_channels[-2] + 512, 512, dropout_rate),
            "upconv3": nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            "dec3": self._decoder_block(encoder_channels[-3] + 256, 256, dropout_rate),
            "upconv2": nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            "dec2": self._decoder_block(encoder_channels[-4] + 128, 128, dropout_rate),
            "upconv1": nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            "dec1": self._decoder_block(encoder_channels[-5] + 64, 64, dropout_rate),
            "final_conv": nn.Conv2d(64, num_classes, kernel_size=1)
        })

        # Warstwa przeskalowująca do wymaganego IMG_SIZE
        self.upsample_to_img_size = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)

    def _decoder_block(self, in_channels, out_channels, dropout_rate=0.1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, bounding_box_mask=None):
        if bounding_box_mask is None:
            bounding_box_mask = torch.zeros_like(x)

        if self.training and self.bounding_box_dropout > 0:
            dropout = nn.Dropout2d(p=self.bounding_box_dropout)
            bounding_box_mask = dropout(bounding_box_mask)

        x = torch.cat([x, bounding_box_mask], dim=1)

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

        final_output = self.upsample_to_img_size(self.decoder["final_conv"](dec1))
        return final_output
