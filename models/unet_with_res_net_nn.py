import torch
import torch.nn as nn
from torchvision import models

class UNetWithResNet(nn.Module):
    def __init__(self, resnet_type="resnet152", num_classes=1):
        super(UNetWithResNet, self).__init__()

        # Wybór ResNet
        if resnet_type == "resnet18":
            resnet = models.resnet18(pretrained=True)
            encoder_channels = [64, 64, 128, 256, 512]  # Kanały dla ResNet18
        elif resnet_type == "resnet34":
            resnet = models.resnet34(pretrained=True)
            encoder_channels = [64, 64, 128, 256, 512]  # Kanały dla ResNet34
        elif resnet_type == "resnet50":
            resnet = models.resnet50(pretrained=True)
            encoder_channels = [64, 256, 512, 1024, 2048]  # Kanały dla ResNet50
        elif resnet_type == "resnet101":
            resnet = models.resnet101(pretrained=True)
            encoder_channels = [64, 256, 512, 1024, 2048]  # Kanały dla ResNet101
        elif resnet_type == "resnet152":
            resnet = models.resnet152(pretrained=True)
            encoder_channels = [64, 256, 512, 1024, 2048]  # Kanały dla ResNet152
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Definicja encodera (z ResNet)
        self.enc1 = nn.Sequential(*list(resnet.children())[:4])  # Pierwsza warstwa ResNet
        self.enc2 = resnet.layer1  # Blok 1 ResNet
        self.enc3 = resnet.layer2  # Blok 2 ResNet
        self.enc4 = resnet.layer3  # Blok 3 ResNet
        self.enc5 = resnet.layer4  # Blok 4 ResNet

        # Bottleneck: automatyczne dostosowanie kanałów
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 1024, kernel_size=3, padding=1),  # Zmniejszenie z max kanałów ResNet
            nn.ReLU(inplace=True)
        )

        # Dekoder (upsampling + skip connections)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._decoder_block(encoder_channels[-2] + 512, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._decoder_block(encoder_channels[-3] + 256, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._decoder_block(encoder_channels[-4] + 128, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._decoder_block(encoder_channels[-5] + 64, 64)

        # Finalna warstwa
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        # Blok dekodera: konwolucje i aktywacje
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(enc5)

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))

        dec3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.upconv1(dec2)
        # Dopasowanie enc1 do wymiarów dec1 przed torch.cat
        enc1_resized = torch.nn.functional.interpolate(enc1, size=dec1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([dec1, enc1_resized], dim=1))

        output = self.final_conv(dec1)

        # Finalne wyjście
        return output

# # Testowanie modelu
# if __name__ == "__main__":
#     # Wybierz typ ResNet
#     resnet_type = "resnet50"  # Możesz zmienić na "resnet18", "resnet34", "resnet101", "resnet152"
#     dcm_path = './../data/0100515c-5204-4f31-98e0-f35e4b00004a.dcm'
#
#     # Utwórz model
#     model = UNetWithResNet(resnet_type=resnet_type, num_classes=1)
#
#     # Testowy obraz wejściowy
#     # x = torch.randn(1, 3, 512, 512)  # Batch size = 1, RGB (3 kanały), rozmiar 256x256
#     dcm = pydicom.dcmread(dcm_path).pixel_array / 255
#     dcm_array = cv2.resize(dcm, (512, 512)).astype(np.float32)
#
#     dcm_tensor = torch.tensor(dcm_array).unsqueeze(0).unsqueeze(0)  # Dodanie wymiaru batcha i kanału
#     dcm_tensor = dcm_tensor.to(torch.float32)
#
#
#
#     y, debug_data = model(dcm_tensor)
#     # visualize_debug_data(debug_data, 'enc1', num_channels=3)
#
#     # Wizualizacja wyjścia enc1 (pierwsza warstwa encodera)
#     visualize_debug_data(debug_data, 'enc1', num_channels=3)
#
#     # Wizualizacja wyjścia enc5 (głębsze cechy z encodera)
#     visualize_debug_data(debug_data, 'enc5', num_channels=3)
#
#     # Wizualizacja wyjścia dec1 (ostatnia warstwa dekodera)
#     visualize_debug_data(debug_data, 'dec1', num_channels=3)

