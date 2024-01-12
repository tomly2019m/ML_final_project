import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, input_size, output_size):
        super(Block, self).__init__()
        self.units = input_size
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, input_size)
        self.fc4 = nn.Linear(input_size, input_size)
        self.theta_f_fc = nn.Linear(input_size, output_size, bias=False)
        self.out = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        theta_f = F.tanh(self.theta_f_fc(x))
        out = self.out(theta_f)
        return out
