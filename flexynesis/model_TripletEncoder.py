# Generating encodings of multi-omic data using triplet loss 
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

from .models_shared import *
from .data import TripletMultiOmicDataset


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
    """
    A PyTorch Lightning module implementing a multi-triplet network with a classifier for multi-modal data.
    
    The network consists of a multi-embedding network that computes embeddings for different input modalities
    and a classifier that takes the concatenated embeddings as input and predicts class labels.

    Attributes:
        multi_embedding_network (MultiEmbeddingNetwork): A multi-embedding network for multiple input modalities.
        classifier (Classifier): A classifier network for predicting class labels from fused embeddings.
    """
    def __init__(self, config, dataset, task = 'classification', val_size = 0.2):
        """
        Initialize the MultiTripletNetwork with the given parameters.

        Args:
            num_layers (int): The number of layers in the multi-embedding network.
            input_sizes (list of int): A list of input sizes for each embedding network in the multi-embedding network.
            hidden_sizes (list of int): A list of hidden sizes for each embedding network in the multi-embedding network.
            output_size (int): The size of the fused embedding output from the multi-embedding network.
            num_classes (int): The number of output classes for the classifier.
        """
        super(MultiTripletNetwork, self).__init__()
        
        self.config = config
        self.task = task
        self.val_size = val_size
        
        layers = list(dataset.dat.keys())
        input_sizes = [len(dataset.features[layers[i]]) for i in range(len(layers))]
        hidden_sizes = [config['hidden_dim'] for x in range(len(layers))]
        num_classes = len(np.unique(dataset.y))
        
        self.multi_embedding_network = MultiEmbeddingNetwork(input_sizes, hidden_sizes, config['embedding_dim'])
        self.classifier = Classifier(config['embedding_dim'] * len(layers), [config['classifier_hidden_dim']], num_classes)
        
        dataset.y = dataset.y.long() # convert double to long for categorical integers
        self.dataset = TripletMultiOmicDataset(dataset) # convert to TripletMultiOmicDataset
        self.dat_train, self.dat_val = self.prepare_data() # split for train/validation
                                                                              
    def forward(self, anchor, positive, negative):
        """
        Compute the forward pass of the MultiTripletNetwork and return the embeddings and predictions.

        Args:
            anchor (dict): A dictionary containing the anchor input tensors for each EmbeddingNetwork.
            positive (dict): A dictionary containing the positive input tensors for each EmbeddingNetwork.
            negative (dict): A dictionary containing the negative input tensors for each EmbeddingNetwork.

        Returns:
            tuple: A tuple containing the anchor, positive, and negative embeddings and the predicted class labels.
        """
        anchor_embedding = self.multi_embedding_network(anchor)
        positive_embedding = self.multi_embedding_network(positive)
        negative_embedding = self.multi_embedding_network(negative)
        y_pred = self.classifier(anchor_embedding)
        return anchor_embedding, positive_embedding, negative_embedding, y_pred
    
    def configure_optimizers(self):
        """
        Configure the optimizer for the MultiTripletNetwork.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        """
        Compute the triplet loss for the given anchor, positive, and negative embeddings.

        Args:
            anchor (torch.Tensor): The anchor embedding tensor.
            positive (torch.Tensor): The positive embedding tensor.
            negative (torch.Tensor): The negative embedding tensor.
            margin (float, optional): The margin for the triplet loss. Default is 1.0.

        Returns:
            torch.Tensor: The computed triplet loss.
        """
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + margin)
        return losses.mean()    

    def training_step(self, train_batch, batch_idx):
        """
        Performs one training step on a batch of data.
        
        Args:
            train_batch (tuple): A tuple containing the embeddings for the anchor, positive, negative samples and the label of the anchor sample.
            batch_idx (int): The index of the current batch.
            
        Returns:
            loss (torch.Tensor): The computed loss for the current training step.
        """
        anchor, positive, negative, y = train_batch[0], train_batch[1], train_batch[2], train_batch[3]
        anchor_embedding, positive_embedding, negative_embedding, y_pred = self.forward(anchor, positive, negative)
        triplet_loss = self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        # compute classifier loss 
        cl_loss = F.cross_entropy(y_pred, y)
        loss = triplet_loss + cl_loss 
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        """
        Performs one validation step on a batch of data.
        
        Args:
            val_batch (tuple): A tuple containing the embeddings for the anchor, positive, negative samples and the label of the anchor sample.
            batch_idx (int): The index of the current batch.
            
        Returns:
            loss (torch.Tensor): The computed loss for the current validation step.
        """
        anchor, positive, negative, y = val_batch[0], val_batch[1], val_batch[2], val_batch[3]
        anchor_embedding, positive_embedding, negative_embedding, y_pred = self.forward(anchor, positive, negative)
        triplet_loss = self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        y_pred = self.classifier(anchor_embedding)
        cl_loss = F.cross_entropy(y_pred, y)
        loss = triplet_loss + cl_loss 
        self.log('val_loss', loss)
        return loss
    
    def prepare_data(self):
        lt = int(len(self.dataset)*(1-self.val_size))
        lv = len(self.dataset)-lt
        dat_train, dat_val = random_split(self.dataset, [lt, lv], 
                                          generator=torch.Generator().manual_seed(42))
        return dat_train, dat_val

    def train_dataloader(self):
        return DataLoader(self.dat_train, batch_size=int(self.config['batch_size']), num_workers=0, pin_memory=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dat_val, batch_size=int(self.config['batch_size']), num_workers=0, pin_memory=True, shuffle=False)    
        
    # dataset: MultiOmicDataset
    def transform(self, dataset):
        """
        Transforms the input dataset by generating embeddings and predictions.
        
        Args:
            dataset (MultiOmicDataset): An instance of the MultiOmicDataset class.
            
        Returns:
            z (pd.DataFrame): A dataframe containing the computed embeddings.
            y_pred (np.ndarray): A numpy array containing the predicted labels.
        """
        self.eval()
        z = pd.DataFrame(self.multi_embedding_network(dataset.dat).detach().numpy())
        z.columns = [''.join(['LF', str(x+1)]) for x in z.columns]
        z.index = dataset.samples
        
        # also return predictions 
        y_pred = self.classifier(self.multi_embedding_network(dataset.dat))
        # convert to labels 
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
        return z, y_pred

    def predict(self, dataset):
        """
        Evaluate the model on a given dataset.

        Args:
            dataset: The dataset to evaluate the model on.

        Returns:
            predicted labels
        """
        self.eval()
        y_pred = self.classifier(self.multi_embedding_network(dataset.dat))
        # convert to labels 
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
        return y_pred
