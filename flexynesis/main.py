import torch 
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

# given a pytorch lightning model and pytorch dataset
def train_model(model, dataset, n_epoch, embedding_size, batch_size, val_size = 0):
    # training
    if val_size == 0:
        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers = 0)
        trainer = pl.Trainer(max_epochs = n_epoch, limit_val_batches = 0, num_sanity_val_steps = 0, 
                             strategy=DDPStrategy(find_unused_parameters=False), num_nodes = 4) 
        trainer.fit(model, train_loader) 
    elif val_size > 0:
        # split train into train/val
        dat_train, dat_val = random_split(dataset, [1-val_size, val_size], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(dat_train, batch_size=batch_size, num_workers = 0)
        val_loader = DataLoader(dat_val, batch_size=batch_size, num_workers = 0)
        trainer = pl.Trainer(max_epochs = n_epoch, 
                             strategy=DDPStrategy(find_unused_parameters=False), 
                             num_nodes = 4) 
        trainer.fit(model, train_loader, val_loader) 
    return model
    