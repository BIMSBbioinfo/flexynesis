# Early fusion Supervised VAE example run

import flexynesis
import pandas as pd
import os
import torch


if __name__ == '__main__':
    # get data
    # output options
    inputDir = '/data/local/buyar/arcas/multiomics_integration/benchmarks/pharmacogx/output/gdsc2_vs_ccle_gex_cnv/100'
    outDir = '.'
    n_epoch = 200
    hidden_dim = 500
    latent_dim = 20
    batch_size = 128
    datatypes = ['layer1', 'layer2']
    drugName = 'Erlotinib'
    torch.set_num_threads(4)

    # import assays and labels
    dat_train = {x: pd.read_csv(os.path.join(inputDir, 'train', ''.join([x, '.csv']))) for x in datatypes}
    dat_holdout = {x: pd.read_csv(os.path.join(inputDir, 'test', ''.join([x, '.holdout.csv']))) for x in datatypes}

    # get drug response data (concatenate to keep all labels in one df)
    drugs = pd.concat([pd.read_csv(os.path.join(inputDir, 'train', 'clin.csv'), sep = '\t').transpose(),
                      pd.read_csv(os.path.join(inputDir, 'test', 'clin.csv'), sep = '\t').transpose()])


    train_dataset = flexynesis.data.make_dataset(dat_train, drugs, drugName, batch_size, concatenate = True)
    holdout_dataset = flexynesis.data.make_dataset(dat_holdout, drugs, drugName, batch_size, concatenate = True)
    
    # define model 
    model = flexynesis.supervised_vae_ef(input_dim = len(train_dataset.features['all']), 
                                 hidden_dim = hidden_dim, latent_dim = latent_dim, num_class = 1, h = 8)
    # train model
    model = flexynesis.train_model(model, train_dataset, n_epoch, batch_size, val_size = 0.2) 

    z_train = model.transform(train_dataset)
    z_holdout = model.transform(holdout_dataset)

    COR = model.evaluate(holdout_dataset)
    stats = pd.DataFrame.from_dict({'RMSE': 'NA', 'Rsquare': 'NA', 'COR': COR, 
                                        'drug': drugName, 'trainSampleN': len(train_dataset), 
                                        'testSampleN': len(holdout_dataset), 
                                        'tool': 'svae_ef'}, orient = 'index').T
    
    # save stats 
    outFile = os.path.join(outDir,  '.'.join(['stats', drugName, 'tsv']))
    print("Saving stats to file", outFile)
    stats.to_csv(outFile, index = False, sep = '\t')