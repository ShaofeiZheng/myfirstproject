import torch
from torch import nn

class Conv_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 128, 3, 1, 1),  # (batch, 2, 4, 100) --> (batch, 128, 4, 100)
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),  # (batch, 128, 4, 100) --> (batch, 64, 4, 100)
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # (batch, 64, 4, 100) --> (batch, 64, 4, 100)
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),  # (batch, 64, 4, 100) --> (batch, 32, 4, 100)
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.output = nn.Linear(12800, 200)  # (batch, 12800) --> (batch, 200)

    def forward(self, F_channel):
        out = self.conv1(F_channel)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        out = self.output(out)
        return out


class PhaseNN(nn.Module):
    def __init__(self):
        super(PhaseNN, self).__init__()
        self.Conv = Conv_Layer()

    def forward(self, F_channel, G_channel, hr_channel):
        # phase transform
        out = self.Conv(F_channel)
        theta1, theta2 = out.chunk(2, dim=1)
        theta_real = torch.cos(theta1)
        theta_imag = torch.sin(theta2)
        Theta = theta_real + 1j * theta_imag

        # calculate the MSE
        D_row, D_col = Theta.size()
        Theta_diag = torch.diag_embed(Theta)
        temp1 = torch.matmul(G_channel, Theta_diag)
        temp2 = torch.matmul(temp1, hr_channel.unsqueeze(2))
        temp3 = torch.norm(temp2.squeeze(2)) ** 2

        return -1 / D_row * temp3



























