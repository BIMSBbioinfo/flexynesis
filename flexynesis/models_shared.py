# Networks that can be reused across different architectures

from torch import nn



# a MLP model for regression/classification
# set num_class to 1 for regression. num_class > 1 => classification
class MLP(nn.Module):
    def __init__(self, num_feature, num_class, h = 32):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(num_feature, h)
        self.layer_out = nn.Linear(h, num_class)
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=0.4)
        self.batchnorm = nn.BatchNorm1d(h)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x