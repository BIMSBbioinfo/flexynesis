import argparse
import os
import yaml
import torch
import pandas as pd
import flexynesis

def main():
    parser = argparse.ArgumentParser(description="Flexynesis - Your PyTorch model training interface")
    
    parser.add_argument("--data_path", help="Path to the folder with train/test data files", type=str)
    parser.add_argument("--model_class", help="The kind of model class to instantiate", type=str, choices=["DirectPred", "supervised_vae", "MultiTripletNetwork"])
    parser.add_argument("--target_variables", help="Which variables in 'clin.csv' to use for predictions", type = str)
    parser.add_argument("--fusion_type", help="How to fuse the omics layers", type=str, choices=["early", "intermediate"])
    parser.add_argument("--hpo_iter", help="Number of iterations for hyperparameter optimisation", type=int)
    parser.add_argument("--features_min", help="Minimum number of features to retain after feature selection", type=int)
    parser.add_argument("--features_top_percentile", help="Top percentile features to retain after feature selection", type=float)
    parser.add_argument("--data_types", help="Which omic data matrices to work on, comma-separated: e.g. 'gex,cnv'", type=str)
    parser.add_argument("--outfile", help="Path to the output file to save the model evaluation stats", type=str)
    
    torch.set_num_threads(4)

    args = parser.parse_args()

    # Validate the data path
    if not os.path.isdir(args.data_path):
        raise ValueError(f"Invalid data_path: {args.data_path}")

    if args.model_class == "DirectPred":
        model_class = flexynesis.DirectPred
        config_name = 'DirectPred'
    elif args.model_class == "supervised_vae":
        model_class = flexynesis.supervised_vae
        config_name = 'SVAE'
    elif args.model_class == "MultiTripletNetwork":
        model_class = flexynesis.MultiTripletNetwork
        config_name = 'MultiTripletNetwork'
    else:
        raise ValueError(f"Invalid model_class: {args.model_class}")

    # import assays and labels
    inputDir = args.data_path
    
    # Set concatenate to True to use early fusion, otherwise it will run intermediate fusion
    concatenate = False 
    if args.fusion_type == 'early':
        concatenate = True
        
    data_importer = flexynesis.DataImporter(path = args.data_path, 
                                            data_types = args.data_types.strip().split(','),
                                            concatenate = concatenate, 
                                            min_features= args.features_min, 
                                            top_percentile= args.features_top_percentile)
    
    train_dataset, test_dataset = data_importer.import_data()
    
    # define a tuner object, which will instantiate a DirectPred class 
    # using the input dataset and the tuning configuration from the config.py
    tuner = flexynesis.HyperparameterTuning(train_dataset, 
                                            model_class = model_class, 
                                            target_variables = args.target_variables.strip().split(','), 
                                            config_name = config_name, 
                                            n_iter=int(args.hpo_iter))    
    
    # do a hyperparameter search training multiple models and get the best_configuration 
    model, best_params = tuner.perform_tuning()
    
    # make predictions on the test dataset
    y_pred = model.predict(test_dataset)
    
    # evaluate predictions 
    if args.task == 'regression':
        stats = flexynesis.utils.evaluate_regressor(test_dataset.y, y_pred)
    if args.task == 'classification':
        stats = flexynesis.utils.evaluate_classifier(test_dataset.y, y_pred)
    
    # save to file 
    pd.DataFrame(stats.items()).transpose().to_csv(args.outfile, header=False, index=False)
    
if __name__ == "__main__":
    main()
