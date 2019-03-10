import torch
import torch.nn as nn

#this model is highly borrowed from https://github.com/liusongxiang/StarGAN-Voice-Conversion/blob/master/model.py

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        # self.layer1 = nn.Sequential(
        #     nn.Linear(1000, 500),
        #     nn.LeakyReLU()
        # )
        #
        # self.layer2 = nn.Sequential(
        #     nn.Linear(500, 200),
        #     nn.LeakyReLU()
        # )
        #
        # self.layer3 = nn.Sequential(
        #     nn.Linear(200, 100),
        #     nn.LeakyReLU()
        # )

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, 3, 1, 1),
            nn.InstanceNorm1d(16, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, 4, 2, padding=1),
            nn.InstanceNorm1d(32, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        #
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )


        #
        # self.


        # self.fc1 =

    def forward(self, input):
        x = input.unsqueeze(1)
        # print(x.shape)
        #input :: 1000 x 1
        x = self.layer1(x)
        # print('... 1', x.size())
        # x = self.max_pool(x)
        # print('.. max pool .. ', x.size())
        x = self.layer2(x)
        # print('... 2', x.size())
        # x = self.max_pool(x)
        x = self.layer3(x)
        # x = self.max_pool(x)

        # x = self.max_pool(x)
        # print('... 3', x.size())
        return x

class Generator(nn.Module):

    def __init__(self, singer):
        super(Generator, self).__init__()
        self.singer = singer

        bottle_neck_layer = []

        for i in range(6):
            bottle_neck_layer.append(ResidualBlock(64, 64))

        self.layer0 = nn.Sequential(*bottle_neck_layer)

        self.layer1 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm1d(32, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm1d(16, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 1, kernel_size=7, stride=1, padding=3, bias=False)
        )

        # self.layer3 = nn.Sequential(
        #     nn.Linear(500, 1000),
        # )
        #
        # self.bn = nn.BatchNorm1d(1)

    def forward(self, input):
        x = self.layer0(input)
        # x = x.squeeze(1)
        x = self.layer1(x)
        # print('G .. layer1', x.size())
        x = self.layer2(x)
        # print('G .. layer2', x.size())
        x = self.layer3(x)
        # print('G .. layer3', x.size())
        # x = nn.Tanh()(x)
        return x.squeeze(1)

class Discriminator(nn.Module):

    def __init__(self, singer, dim=32):
        super(Discriminator, self).__init__()
        self.singer = singer

        layers = []
        layers.append(nn.Conv1d(1, dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = dim

        for i in range(5):
            layers.append(nn.Conv1d(curr_dim, 2 * curr_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim *= 2

        layers.append(nn.Conv1d(curr_dim, 1, kernel_size=1, stride=1))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        #batch_size * 1 * 1024
        x = self.main(input).squeeze(2)
        return x #size, batch_size * 1

