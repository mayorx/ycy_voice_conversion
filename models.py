import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(500, 200),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(200, 100),
            nn.LeakyReLU()
        )

        # self.layer1 = nn.Sequential(
        #     nn.Conv1d(1, 16, 21, 1, 0),
        #     nn.BatchNorm1d(16),
        #     nn.LeakyReLU()
        # )
        #
        # self.max_pool = nn.MaxPool1d(2)
        #
        # self.layer2 = nn.Sequential(
        #     nn.Conv1d(16, 32, 11, 1, 0),
        #     nn.BatchNorm1d(32),
        #     nn.LeakyReLU()
        # )
        #
        # self.layer3 = nn.Sequential(
        #     nn.Conv1d(32, 64, 21, 1, 0),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU()
        # )
        #
        # self.


        # self.fc1 =

    def forward(self, input):
        #input :: 1000 x 1
        x = self.layer1(input)
        # print('... 1', x.size())
        # x = self.max_pool(x)
        # print('.. max pool .. ', x.size())
        x = self.layer2(x)
        # print('... 2', x.size())
        x = self.layer3(x)
        # x = self.max_pool(x)
        # print('... 3', x.size())
        return x



class Generator(nn.Module):

    def __init__(self, singer):
        super(Generator, self).__init__()
        self.singer = singer

        self.layer1 = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(200, 500),
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(500, 1000),
        )

        self.bn = nn.BatchNorm1d(1)

    def forward(self, input):
        # x = self.bn(input.unsqueeze(1))
        # x = x.squeeze(1)
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = nn.Tanh()(x)
        return x


class Discriminator():
    def __init__(self, singer):
        pass
