# Early fusion Supervised VAE example run

import flexynesis
import pandas as pd
import os
import torch
import numpy as np

# filter features to keep top 20% based on laplacian score, keeping minimum of 1000 features
# TODO: make this configurable
def filter(dat):
    counts = {x: max(int(dat[x].shape[0]/20), 1000) for x in dat.keys()}
    dat = {x: flexynesis.filter_by_laplacian(dat[x].T, topN=counts[x]).T for x in dat.keys()}
    return dat

# subset dat2 to only include features available in dat1
def harmonize(dat1, dat2):
    # features to keep
    features = {x: dat1[x].index for x in dat1.keys()}
    return {x: dat2[x].loc[features[x]] for x in dat2.keys()}

# label numeric values by position with the median
def label_by_median(arr):
    arr = np.asarray(arr)
    median = np.median(arr)
    categories = np.where(arr > median, 0, 1) 
    return categories


if __name__ == '__main__':
    # get data
    # output options
    inputDir = '/data/local/buyar/arcas/multiomics_integration/benchmarks/pharmacogx/output/gdsc2_vs_ccle_gex_cnv/100'
    outDir = '.'
    n_epoch = 300
    batch_size = 64
    datatypes = ['layer1', 'layer2']
    drugName = 'Erlotinib'
    torch.set_num_threads(4)

    # import assays and labels
    dat_train = {x: pd.read_csv(os.path.join(inputDir, 'train', ''.join([x, '.csv']))) for x in datatypes}
    dat_holdout = {x: pd.read_csv(os.path.join(inputDir, 'test', ''.join([x, '.holdout.csv']))) for x in datatypes}

    # get drug response data (concatenate to keep all labels in one df)
    drugs = pd.concat([pd.read_csv(os.path.join(inputDir, 'train', 'clin.csv'), sep = '\t').transpose(),
                      pd.read_csv(os.path.join(inputDir, 'test', 'clin.csv'), sep = '\t').transpose()])

    # skip filtering
    #dat_train = filter(dat_train)
    #dat_holdout = harmonize(dat_train, dat_holdout)
    
    # Set concatenate to True to use early fusion, otherwise it will run intermediate fusion
    train_dataset = flexynesis.make_dataset(dat_train, drugs, drugName, batch_size, concatenate = False)
    holdout_dataset = flexynesis.make_dataset(dat_holdout, drugs, drugName, batch_size, concatenate = False)
    # convert numeric values to categorical 
    train_dataset.y = torch.tensor(label_by_median(train_dataset.y))
    holdout_dataset.y = torch.tensor(label_by_median(holdout_dataset.y))
    
    # create triplet datasets
    # get triplets
    triplet_train_dataset = flexynesis.TripletMultiOmicDataset(train_dataset)
    triplet_holdout_dataset = flexynesis.TripletMultiOmicDataset(holdout_dataset, train=False)
    
    # define triplet network
    layers = list(train_dataset.dat.keys())
    input_dims = [len(train_dataset.features[layers[i]]) for i in range(len(layers))]
    hidden_dims = [256, 256]
    output_dim = 32
    # define model 
    multi_triplet_network = flexynesis.MultiTripletNetwork(num_layers = len(layers), input_sizes = input_dims, 
                                                           hidden_sizes = hidden_dims, 
                                                           output_size = output_dim, 
                                                           num_classes=len(triplet_train_dataset.labels_set))    
    # train model
    model = flexynesis.train_model(multi_triplet_network, triplet_train_dataset, n_epoch, batch_size, val_size = 0.2) 

    z_train, y_pred_train = model.transform(train_dataset)
    z_holdout, y_pred_holdout = model.transform(holdout_dataset)
    
    z_train['y'] = train_dataset.y
    z_holdout['y'] = holdout_dataset.y
    
    z_train.to_csv('z_train.csv')
    z_holdout.to_csv('z_holdout.csv')
    
    # evaluate
    print("train stats")
    train_stats = flexynesis.utils.evaluate_classifier(train_dataset.y, y_pred_train)
    print("holdout stats")    
    holdout_stats = flexynesis.utils.evaluate_classifier(holdout_dataset.y, y_pred_holdout)
    