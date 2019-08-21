import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, img_size, z_dim, initial_channel):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.initial_channel = initial_channel
        layers = []
        layer1 = []
        convblock = []
        if img_size == 64:
            self.repeat_num = 4
        if img_size == 128:
            self.repeat_num = 5

        self.fc = nn.Linear(self.z_dim, self.initial_channel*8*8)#128->128*8*8

        layer1.append(nn.Conv2d(in_channels=self.initial_channel, out_channels=self.initial_channel, kernel_size=3, padding=1, stride=1))
        layer1.append(nn.ELU(True))
        layer1.append(nn.Conv2d(self.initial_channel, self.initial_channel, 3, 1, 1))
        layer1.append(nn.ELU(True))

        for indx in range(self.repeat_num):
            layers.append(layer1)
            layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        layers = layers.append(nn.Conv2d(in_channels=self.initial_channel,
                                                 out_channels=3,
                                                 kernel_size=3,
                                                padding=1, stride=1))
        self.convblock = layers.append(nn.ELU(True))


    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.initial_channel, 8, 8)
        out = self.convblock(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, img_size, z_dim, initial_channels):
        super(Discriminator, self).__init__()
        self .z_dim = z_dim
        self.initinal_channels = initial_channels
        if img_size == 64:
            self.repeat_num = 4
        if img_size == 128:
            self.repeat_num = 5

        #Encoder
        # for idx in range(self.repeat_num):
        self.conv = nn.Sequential(
                   nn.Conv2d(3, self.initinal_channels, 3, 1, 1),
                   nn.ELU(True),
                   nn.Conv2d(self.initial_channels, self.initial_channels, 3, 1, 1),
                   nn.ELU(True),
                   nn.Conv2d(self.initial_channels, self.initial_channels*2, 3, 2, 1),
                   nn.ELU(True),
                   nn.Conv2d(self.initial_channels*2, self.initial_channels*2, 3, 1, 1),
                   nn.ELU(True),
                   nn.Conv2d(self.initial_channels*2, self.initial_channels*3, 3, 2, 1),
                   nn.ELU(True),
                   nn.Conv2d(self.initial_channels*3, self.initial_channels*3, 3, 1, 1),
                   nn.ELU(True),
                   nn.Conv2d(self.initial_channels*3, self.initial_channels*3, 3, 1, 1),
                   nn.ELU(True),
                )
        self.fc1 = nn.Linear(8*8*self.initial_channels*3, self.z_dim)
        
        #Decoder
        self.fc2 = nn.Linear(self.z_dim, 8 * 8 * self.initial_channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.initial_channels, self.initial_channels, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(self.initial_channels, self.initial_channels, 3, 1, 1),
            nn.ELU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.initial_channels, self.initial_channels, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(self.initial_channels, self.initial_channels, 3, 1, 1),
            nn.ELU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.initial_channels, self.initial_channels, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(self.initial_channels, self.initial_channels, 3, 1, 1),
            nn.ELU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.initial_channels, 3, 3, 1, 1),
            nn.Tanh(),
            #nn.ELU(True)
        )

    def forward(self, input):
        #encode
        encoder_output = self.conv(input)
        latent_code = self.fc1(encoder_output)

        #decode
        latent_resize = self.fc2(latent_code).view(input.size(0), self.initinal_channels, 8, 8)
        out = self.conv1(latent_resize)
        out = F.upsample(out, scale_factor=2, mode='nearest')
        out = self.conv2(out)
        out = F.upsample(out, scale_factor=2, mode='nearest')
        out = self.conv3(out)
        out = F.upsample(out, scale_factor=2, mode='nearest')
        out = self.conv4(out)

        return out

