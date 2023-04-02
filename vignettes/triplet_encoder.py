# Early fusion Supervised VAE example run

import flexynesis
import pandas as pd
import os
import torch
import numpy as np


if __name__ == '__main__':
    torch.set_num_threads(4)

    # import data, pick top 10% features (min 500), build a model for Erlotinib
    data_importer = flexynesis.DataImporter(path = '/data/local/buyar/arcas/multiomics_integration/datasets/gdsc_vs_ccle/', 
                                            outcome_var = 'Erlotinib', 
                                            data_types = ['gex', 'cnv'],
                                            min_features = 500, 
                                            top_percentile = 0.2, 
                                            convert_to_labels = True)
    
    train_dataset, test_dataset = data_importer.import_data()
    
    # augment dataset using PCA distortion
    train_dataset = flexynesis.augment_dataset_with_pc_distortion(train_dataset, [0, 1, 2], [0.8, 1.2], 2)
    
    triplet_train_dataset = flexynesis.TripletMultiOmicDataset(train_dataset)
    triplet_test_dataset = flexynesis.TripletMultiOmicDataset(test_dataset, train=False)
    
    # define tuning 
    tuner = flexynesis.HyperparameterTuning(train_dataset, model_class = flexynesis.MultiTripletNetwork, 
                                            config_name = 'MultiTripletNetwork', n_iter=20)
    
    # train model and get best model
    model, best_params = tuner.perform_tuning()
    
    z_train, y_pred_train = model.transform(train_dataset)
    z_test, y_pred_test = model.transform(test_dataset)
        
    # evaluate
    print("train stats")
    train_stats = flexynesis.utils.evaluate_classifier(train_dataset.y, y_pred_train)
    print("test stats")    
    test_stats = flexynesis.utils.evaluate_classifier(test_dataset.y, y_pred_test)
    