import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, input_size, output_size, seq_length=96):
        super(Block, self).__init__()
        self.units = input_size
        self.thetas_dim = output_size
        self.seq_length = seq_length
        self.fc1 = nn.Linear(seq_length, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, input_size)
        self.fc4 = nn.Linear(input_size, input_size)

        self.theta_b_fc = nn.Linear(input_size, output_size, bias=False)
        self.theta_f_fc = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
