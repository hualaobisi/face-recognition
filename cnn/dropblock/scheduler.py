import numpy as np
from torch import nn


class LinearScheduler(nn.Module):
    def __init__(self, droptype, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.droptype = droptype
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.droptype(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.droptype.drop_prob = self.drop_values[self.i]

        self.i += 1
