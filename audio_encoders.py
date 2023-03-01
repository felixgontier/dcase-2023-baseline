### Adapted from task 6b baseline xieh97/dcase2023-audio-retrieval

import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class CNN14Encoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(CNN14Encoder, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.cnn = nn.Sequential(
            # Conv2D block1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            # Conv2D block2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            # Conv2D block3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            # Conv2D block4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            # Conv2D block5
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            # Conv2D block6
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(p=0.2)
        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Linear(2048, kwargs["out_dim"], bias=True)

        self.bn0.apply(init_weights)
        self.cnn.apply(init_weights)
        self.fc.apply(init_weights)
        self.fc2.apply(init_weights)

    def forward(self, x, skip_fc=False):
        """
        :param x: tensor, (batch_size, time_steps, Mel_bands).
        :return: tensor, (batch_size, embed_dim).
        """
        x = x.unsqueeze(1)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.cnn(x)
        x = torch.mean(x, dim=3) # (N, 2048, T/64)
        
        if skip_fc:
            return x.transpose(1,2) # b,t,h
        else:
            (x1, _) = torch.max(x, dim=2)  # max across time
            x2 = torch.mean(x, dim=2)  # average over time
            x = x1 + x2  # (N, 2048)

            x = self.fc(x)  # (N, 2048)
            x = self.fc2(x)  # (N, embed_dim)

            return x
