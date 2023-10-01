import torch
from torch import nn

from .models_shared import Encoder, Decoder, MLP, EmbeddingNetwork, Classifier
from .model_TripletEncoder import MultiEmbeddingNetwork

__all__ = ["Encoder", "Decoder", "MLP", "EmbeddingNetwork", "MultiEmbeddingNetwork", "Classifier", "CNN"]


class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.layer_out = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        """(N, C) -> (N, C, L) -> (N, C).
        """
        x = x.unsqueeze(-1)

        x = self.layer_1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        x = x.squeeze(-1)
        return x
