from pytorch_lightning import seed_everything
# Set the seed for all the possible random number generators.
seed_everything(42, workers=True)
import argparse
from typing import NamedTuple
import os
import yaml
import torch
import pandas as pd
import flexynesis
from flexynesis.models import *
import warnings

def main():
    parser = argparse.ArgumentParser(description="Flexynesis - Your PyTorch model training interface", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--data_path", help="(Required) Path to the folder with train/test data files", type=str, required = True)
    parser.add_argument("--model_class", help="(Required) The kind of model class to instantiate", type=str, choices=["DirectPred", "DirectPredCNN", "DirectPredGCNN", "supervised_vae", "MultiTripletNetwork"], required = True)
    parser.add_argument("--target_variables", help="(Required) Which variables in 'clin.csv' to use for predictions, comma-separated if multiple", type = str, required = True)
    parser.add_argument('--config_path', type=str, default=None, help='Optional path to an external hyperparameter configuration file in YAML format.')
    parser.add_argument("--batch_variables", 
                        help="(Optional) Which variables in 'clin.csv' to use for data integration / batch correction, comma-separated if multiple", 
                        type = str, default = None)
    parser.add_argument("--fusion_type", help="How to fuse the omics layers", type=str, choices=["early", "intermediate"], default = 'intermediate')
    parser.add_argument("--hpo_iter", help="Number of iterations for hyperparameter optimisation", type=int, default = 5)
    parser.add_argument("--features_min", help="Minimum number of features to retain after feature selection", type=int, default = 500)
    parser.add_argument("--features_top_percentile", help="Top percentile features to retain after feature selection", type=float, default = 0.2)
    parser.add_argument("--data_types", help="(Required) Which omic data matrices to work on, comma-separated: e.g. 'gex,cnv'", type=str, required = True)
    parser.add_argument("--outdir", help="Path to the output folder to save the model outputs", type=str, default = os.getcwd())
    parser.add_argument("--prefix", help="Job prefix to use for output files", type=str, default = 'job')
    parser.add_argument("--log_transform", help="whether to apply log-transformation to input data matrices", type=str, choices=['True', 'False'], default = 'False')
    parser.add_argument("--threads", help="Number of threads to use", type=int, default = 4)
    parser.add_argument("--early_stop_patience", help="How many epochs to wait when no improvements in validation loss is observed (default: -1; no early stopping)", type=int, default = -1)
    parser.add_argument("--use_loss_weighting", help="whether to apply loss-balancing using uncertainty weights method", type=str, choices=['True', 'False'], default = 'True')
    parser.add_argument("--evaluate_baseline_performance", help="whether to run Random Forest + SVMs to see the performance of off-the-shelf tools on the same dataset", type=str, choices=['True', 'False'], default = 'True')

    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", "has been removed as a dependency of the")
    warnings.filterwarnings("ignore", "The `srun` command is available on your system but is not used")

    args = parser.parse_args()
    
    torch.set_num_threads(args.threads)

    # Validate paths
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Input --data_path doesn't exist at:",  {args.data_path})
    if not os.path.exists(args.outdir):
        raise FileNotFoundError(f"Path to --outdir doesn't exist at:",  {args.outdir})

    class AvailableModels(NamedTuple):
        # type AvailableModel = ModelClass: Type, ModelConfig: str
        DirectPred: tuple[DirectPred, str] = DirectPred, "DirectPred"
        supervised_vae: tuple[supervised_vae, str] = supervised_vae, "supervised_vae"
        MultiTripletNetwork: tuple[MultiTripletNetwork, str] = MultiTripletNetwork, "MultiTripletNetwork"
        DirectPredCNN: tuple[DirectPredCNN, str] = DirectPredCNN, "DirectPredCNN"
        DirectPredGCNN: tuple[DirectPredGCNN, str] = DirectPredGCNN, "DirectPredGCNN"

    available_models = AvailableModels()
    model_class = getattr(available_models, args.model_class, None)
    if model_class is None:
        raise ValueError(f"Invalid model_class: {args.model_class}")
    else:
        model_class, config_name = model_class

    # Set use_graph var
    use_graph = True if config_name == "DirectPredGCNN" else False

    # import assays and labels
    inputDir = args.data_path
    
    # Set concatenate to True to use early fusion, otherwise it will run intermediate fusion
    concatenate = False 
    if args.fusion_type == 'early':
        concatenate = True
        
    data_importer = flexynesis.DataImporter(path = args.data_path, 
                                            data_types = args.data_types.strip().split(','),
                                            concatenate = concatenate, 
                                            log_transform = args.log_transform == 'True',
                                            min_features= args.features_min, 
                                            top_percentile= args.features_top_percentile,
                                            use_graph=use_graph,
    )
    
    train_dataset, test_dataset = data_importer.import_data()
    
    # define a tuner object, which will instantiate a DirectPred class 
    # using the input dataset and the tuning configuration from the config.py
    tuner = flexynesis.HyperparameterTuning(train_dataset, 
                                            model_class = model_class, 
                                            target_variables = args.target_variables,
                                            batch_variables = args.batch_variables,
                                            config_name = config_name, 
                                            config_path = args.config_path,
                                            n_iter=int(args.hpo_iter),
                                            use_loss_weighting = args.use_loss_weighting == 'True',
                                            early_stop_patience = int(args.early_stop_patience))    
    
    # do a hyperparameter search training multiple models and get the best_configuration 
    model, best_params = tuner.perform_tuning()
        
    # evaluate predictions 
    print("Computing model evaluation metrics")
    metrics_df = flexynesis.evaluate_wrapper(model.predict(test_dataset), test_dataset)
    metrics_df.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'stats.csv'])), header=True, index=False)
    
    # print known/predicted labels 
    predicted_labels = pd.concat([flexynesis.get_predicted_labels(model.predict(train_dataset), train_dataset, 'train'),
                                  flexynesis.get_predicted_labels(model.predict(test_dataset), test_dataset, 'test')], 
                                ignore_index=True)
    predicted_labels.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'predicted_labels.csv'])), header=True, index=False)
    # compute feature importance values
    print("Computing variable importance scores")
    for var in model.target_variables:
        model.compute_feature_importance(var, steps = 20)
    df_imp = pd.concat([model.feature_importances[x] for x in model.target_variables], 
                       ignore_index = True)
    df_imp.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'feature_importance.csv'])), header=True, index=False)

    # get sample embeddings and save 
    print("Extracting sample embeddings")
    embeddings_train = model.transform(train_dataset)
    embeddings_test = model.transform(test_dataset)
    
    embeddings_train.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'embeddings_train.csv'])), header=True)
    embeddings_test.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'embeddings_test.csv'])), header=True)
    
    # also filter embeddings to remove batch-associated dims and only keep target-variable associated dims 
    print("Printing filtered embeddings")
    embeddings_train_filtered = flexynesis.remove_batch_associated_variables(data = embeddings_train, 
                                                                             batch_dict={x: train_dataset.ann[x] for x in model.batch_variables} if model.batch_variables is not None else None, 
                                                                             target_dict={x: train_dataset.ann[x] for x in model.target_variables}, 
                                                                             variable_types=train_dataset.variable_types)
    # filter test embeddings to keep the same dims as the filtered training embeddings
    embeddings_test_filtered = embeddings_test[embeddings_train_filtered.columns]

    # save 
    embeddings_train_filtered.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'embeddings_train.filtered.csv'])), header=True)    
    embeddings_test_filtered.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'embeddings_test.filtered.csv'])), header=True)    
    
    # evaluate off-the-shelf methods on the main target variable 
    if args.evaluate_baseline_performance == 'True':
        print("Computing off-the-shelf method performance on first target variable:",model.target_variables[0])
        metrics_baseline = flexynesis.evaluate_baseline_performance(train_dataset, test_dataset, 
                                                                    variable_name= model.target_variables[0], 
                                                                    n_folds=5)
        metrics_baseline.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'baseline.stats.csv'])), header=True, index=False) 
    
    # save the trained model in file
    torch.save(model, os.path.join(args.outdir, '.'.join([args.prefix, 'final_model.pth'])))
    
if __name__ == "__main__":
    main()
