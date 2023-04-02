# Early fusion Supervised VAE example run

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
                                            min_features = 500, 
                                            top_percentile = 0.1, 
                                            convert_to_labels = False)
    
    train_dataset, test_dataset = data_importer.import_data()
    
    # augment dataset using PCA distortion
    train_dataset = flexynesis.augment_dataset_with_pc_distortion(train_dataset, [0, 1, 2], [0.8, 1.2], 1)
    
    # define tuning 
    tuner = flexynesis.HyperparameterTuning(train_dataset, model_class = flexynesis.supervised_vae, 
                                            config_name = 'SVAE', n_iter=1)
    
    # train model and get best model
    model, best_params = tuner.perform_tuning()
    
    # export latent factors and evaluate the model 
    # z_train = model.transform(train_dataset)
    # z_test = model.transform(test_dataset)

    COR = model.evaluate(test_dataset)
    stats = pd.DataFrame.from_dict({'RMSE': 'NA', 'Rsquare': 'NA', 'COR': COR, 
                                    'trainSampleN': len(train_dataset), 
                                    'testSampleN': len(test_dataset), 
                                    'tool': 'svae'}, orient = 'index').T
    
    print(stats)