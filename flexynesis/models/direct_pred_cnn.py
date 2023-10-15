import numpy as np
from torch import nn

from .direct_pred import DirectPred
from ..modules import CNN


class DirectPredCNN(DirectPred):
    def _init_encoders(self):
        layers = list(self.dataset.dat.keys())
        input_dims = [len(self.dataset.features[layers[i]]) for i in range(len(layers))]
        self.encoders = nn.ModuleList([
            CNN(input_dim=input_dims[i], hidden_dim=self.config["hidden_dim"], output_dim=self.config["latent_dim"])
            for i in range(len(layers))
        ])

    def _init_output_layers(self):
        layers = list(self.dataset.dat.keys())
        self.MLPs = nn.ModuleDict()
        for var in self.target_variables:
            if self.dataset.variable_types[var] == "numerical":
                num_class = 1
            else:
                num_class = len(np.unique(self.dataset.ann[var]))
            self.MLPs[var] = CNN(
                input_dim=self.config["latent_dim"] * len(layers),
                hidden_dim=self.config["hidden_dim"],
                output_dim=num_class,
            )
