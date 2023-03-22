# Generating encodings of multi-omic data using triplet loss 
import torch
from torch import nn
from torch.nn import functional as F

import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

from .models_shared import *


class MultiEmbeddingNetwork(nn.Module):
    """
    A neural network module that computes multiple embeddings and fuses them into a single representation.

    Attributes:
        embedding_networks (nn.ModuleList): A list of EmbeddingNetwork instances for each input feature.
    """
    def __init__(self, input_sizes, hidden_sizes, output_size):
        """
        Initialize the MultiEmbeddingNetwork with the given input sizes, hidden sizes, and output size.

        Args:
            input_sizes (list of int): A list of input sizes for each EmbeddingNetwork instance.
            hidden_sizes (list of int): A list of hidden sizes for each EmbeddingNetwork instance.
            output_size (int): The size of the fused embedding output.
        """
        super(MultiEmbeddingNetwork, self).__init__()
        self.embedding_networks = nn.ModuleList([
            EmbeddingNetwork(input_size, hidden_size, output_size)
            for input_size, hidden_size in zip(input_sizes, hidden_sizes)
        ])

    def forward(self, x):
        """
        Compute the forward pass of the MultiEmbeddingNetwork and return the fused embedding.

        Args:
            x (dict): A dictionary containing the input tensors for each EmbeddingNetwork.
                      Keys should correspond to the feature names and values to the input tensors.
        
        Returns:
            torch.Tensor: The fused embedding tensor resulting from the concatenation of individual embeddings.
        """
        embeddings = [
            embedding_network(x[key])
            for key, embedding_network in zip(x.keys(), self.embedding_networks)
        ]
        fused_embedding = torch.cat((embeddings), dim=-1)
        return fused_embedding

class MultiTripletNetwork(pl.LightningModule):
    def __init__(self, num_layers, input_sizes, hidden_sizes, output_size, num_classes):
        super(MultiTripletNetwork, self).__init__()
        self.multi_embedding_network = MultiEmbeddingNetwork(input_sizes, hidden_sizes, output_size)
        self.classifier = Classifier(output_size * num_layers, [64], num_classes)

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.multi_embedding_network(anchor)
        positive_embedding = self.multi_embedding_network(positive)
        negative_embedding = self.multi_embedding_network(negative)
        y_pred = self.classifier(anchor_embedding)
        return anchor_embedding, positive_embedding, negative_embedding, y_pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + margin)
        return losses.mean()    

    def training_step(self, train_batch, batch_idx):
        anchor, positive, negative, y = train_batch[0], train_batch[1], train_batch[2], train_batch[3]
        anchor_embedding, positive_embedding, negative_embedding, y_pred = self.forward(anchor, positive, negative)
        triplet_loss = self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        # compute classifier loss 
        cl_loss = F.cross_entropy(y_pred, y)
        loss = triplet_loss + cl_loss 
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        anchor, positive, negative, y = val_batch[0], val_batch[1], val_batch[2], val_batch[3]
        anchor_embedding, positive_embedding, negative_embedding, y_pred = self.forward(anchor, positive, negative)
        triplet_loss = self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        y_pred = self.classifier(anchor_embedding)
        cl_loss = F.cross_entropy(y_pred, y)
        loss = triplet_loss + cl_loss 
        self.log('val_loss', loss)
        return loss
    
    # dataset: MultiOmicDataset
    def transform(self, dataset):
        self.eval()
        z = pd.DataFrame(self.multi_embedding_network(dataset.dat).detach().numpy())
        z.columns = [''.join(['LF', str(x+1)]) for x in z.columns]
        z.index = dataset.samples
        
        # also return predictions 
        y_pred = self.classifier(self.multi_embedding_network(dataset.dat))
        # convert to labels 
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
        return z, y_pred

