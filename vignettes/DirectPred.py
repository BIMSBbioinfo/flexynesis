# Late integration for multi-omics (limo): like moli without triplet loss

import flexynesis
import pandas as pd
import os
import torch


if __name__ == '__main__':
    torch.set_num_threads(4)

    # import data, pick top 10% features (min 500), build a model for Erlotinib
    data_importer = flexynesis.DataImporter(path = '/data/local/buyar/arcas/multiomics_integration/datasets/gdsc_vs_ccle/', 
                                            outcome_var = 'Erlotinib', 
                                            data_types = ['gex', 'cnv'],
                                            min_features= 500, 
                                            top_percentile=0.1, 
                                            convert_to_labels=False)
    
    train_dataset, test_dataset = data_importer.import_data()
    
    # augment dataset using PCA distortion
    train_dataset = flexynesis.augment_dataset_with_pc_distortion(train_dataset, [0, 1, 2], [0.8, 1.2], 5)
    
    # define a tuner object, which will instantiate a DirectPred class using the input dataset and the tuning configuration from the config.py
    tuner = flexynesis.HyperparameterTuning(train_dataset, model_class = flexynesis.DirectPred, 
                                            config_name = 'DirectPred', n_iter=1)    
    
    # do a hyperparameter search training multiple models and get the best_configuration 
    model, best_params = tuner.perform_tuning()

    # evaluate the model on holdout dataset
    COR = model.evaluate(test_dataset)
    stats = pd.DataFrame.from_dict({'RMSE': 'NA', 'Rsquare': 'NA', 'COR': COR, 
                                        'trainSampleN': len(train_dataset), 
                                        'testSampleN': len(test_dataset), 
                                        'tool': 'DirectPred'}, orient = 'index').T
    print(stats)
    