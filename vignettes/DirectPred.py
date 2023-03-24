# Late integration for multi-omics (limo): like moli without triplet loss

import flexynesis
import pandas as pd
import os
import torch


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

if __name__ == '__main__':
    # get data
    # output options
    inputDir = '/data/local/buyar/arcas/multiomics_integration/benchmarks/pharmacogx/output/gdsc2_vs_ccle_gex_cnv/100'
    outDir = '.'
    datatypes = ['layer1', 'layer2']
    drugName = 'Erlotinib'
    torch.set_num_threads(4)

    # import assays and labels
    dat_train = {x: pd.read_csv(os.path.join(inputDir, 'train', ''.join([x, '.csv']))) for x in datatypes}
    dat_holdout = {x: pd.read_csv(os.path.join(inputDir, 'test', ''.join([x, '.holdout.csv']))) for x in datatypes}

    # get drug response data (concatenate to keep all labels in one df)
    drugs = pd.concat([pd.read_csv(os.path.join(inputDir, 'train', 'clin.csv'), sep = '\t').transpose(),
                      pd.read_csv(os.path.join(inputDir, 'test', 'clin.csv'), sep = '\t').transpose()])

    dat_train = filter(dat_train)
    dat_holdout = harmonize(dat_train, dat_holdout)

    # Set concatenate to True to use early fusion, otherwise it will run intermediate fusion
    train_dataset = flexynesis.data.make_dataset(dat_train, drugs, drugName, concatenate = False)
    holdout_dataset = flexynesis.data.make_dataset(dat_holdout, drugs, drugName, concatenate = False)
    
    # define a tuner object, which will instantiate a DirectPred class using the input dataset and the tuning configuration from the config.py
    tuner = flexynesis.HyperparameterTuning(train_dataset, model_class = flexynesis.DirectPred, 
                                            config_name = 'DirectPred', n_iter=50)    
    
    # do a hyperparameter search training multiple models and get the best_configuration 
    model, best_params = tuner.perform_tuning()

    # evaluate the model on holdout dataset
    COR = model.evaluate(holdout_dataset)
    stats = pd.DataFrame.from_dict({'RMSE': 'NA', 'Rsquare': 'NA', 'COR': COR, 
                                        'drug': drugName, 'trainSampleN': len(train_dataset), 
                                        'testSampleN': len(holdout_dataset), 
                                        'tool': 'DirectPred'}, orient = 'index').T
    
    # save stats 
    outFile = os.path.join(outDir,  '.'.join(['stats', drugName, 'tsv']))
    print("Saving stats to file", outFile)
    stats.to_csv(outFile, index = False, sep = '\t')
