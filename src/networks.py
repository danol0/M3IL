import torch
import torch.nn as nn
import math

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


# Omics
class MaxNet(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, dropout_rate=0.25):
        super(MaxNet, self).__init__()
        hidden = [64, 48, 32, omic_dim]
        # hidden = [80, 120, 80, omic_dim]

        layers = []
        for i in range(len(hidden)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden[i-1], hidden[i]))
            layers.append(nn.ELU())
            layers.append(nn.AlphaDropout(p=dropout_rate, inplace=False))

        self.encoder = nn.Sequential(*layers)
        self.grade_clf = nn.Sequential(nn.Linear(omic_dim, 3))
        self.survival_clf = nn.Sequential(nn.Linear(omic_dim, 1))

        self.register_buffer('output_range', torch.FloatTensor([6]))
        self.register_buffer('output_shift', torch.FloatTensor([-3]))

        self.LSM = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        features = self.encoder(x)
        grade = self.LSM(self.grade_clf(features))
        survival = self.sigmoid(self.survival_clf(features)) * self.output_range + self.output_shift

        return features, grade, survival
