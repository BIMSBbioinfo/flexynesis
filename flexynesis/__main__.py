from lightning import seed_everything
import lightning as pl
from typing import NamedTuple
import os, yaml, torch, time, random, warnings, argparse 
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import flexynesis
from flexynesis.models import *
from lightning.pytorch.callbacks import EarlyStopping
from .data import STRING, MultiOmicDatasetNW

def main():
    """
    Main function to parse command-line arguments and initiate the training interface for PyTorch models.

    This function sets up argument parsing for various parameters required to train and evaluate a PyTorch model,
    including data paths, model class, hyperparameters, and configuration options.

    Args:
        --data_path (str): Path to the folder with train/test data files. (Required)
        --model_class (str): The kind of model class to instantiate. Choices are ["DirectPred", "GNN", "supervised_vae", "MultiTripletNetwork", "CrossModalPred"]. (Required)
        --gnn_conv_type (str): If model_class is set to GNN, choose which graph convolution type to use. Choices are ["GC", "GCN", "SAGE"].
        --target_variables (str): Which variables in 'clin.csv' to use for predictions, comma-separated if multiple. Optional if survival variables are not set to None.
        --batch_variables (str): Which variables in 'clin.csv' to use for data integration/batch correction, comma-separated if multiple. Optional.
        --surv_event_var (str): Which column in 'clin.csv' to use as event/status indicator for survival modeling.
        --surv_time_var (str): Which column in 'clin.csv' to use as time/duration indicator for survival modeling.
        --config_path (str): Optional path to an external hyperparameter configuration file in YAML format.
        --fusion_type (str): How to fuse the omics layers. Choices are ["early", "intermediate"]. Default is 'intermediate'.
        --hpo_iter (int): Number of iterations for hyperparameter optimisation. Default is 5.
        --finetuning_samples (int): Number of samples from the test dataset to use for fine-tuning the model. Set to 0 to disable fine-tuning. Default is 0.
        --variance_threshold (float): Variance threshold (as percentile) to drop low variance features. Default is 1; set to 0 for no variance filtering.
        --correlation_threshold (float): Correlation threshold to drop highly redundant features. Default is 0.8; set to 1 for no redundancy filtering.
        --restrict_to_features (str): Restrict the analysis to the list of features provided by the user. Default is None.
        --subsample (int): Downsample training set to randomly drawn N samples for training. Disabled when set to 0. Default is 0.
        --features_min (int): Minimum number of features to retain after feature selection. Default is 500.
        --features_top_percentile (float): Top percentile features (among the features remaining after variance filtering and data cleanup) to retain after feature selection. Default is 20.
        --data_types (str): Which omic data matrices to work on, comma-separated (e.g., 'gex,cnv'). (Required)
        --input_layers (str): If model_class is set to CrossModalPred, choose which data types to use as input/encoded layers, comma-separated if multiple.
        --output_layers (str): If model_class is set to CrossModalPred, choose which data types to use as output/decoded layers, comma-separated if multiple.
        --outdir (str): Path to the output folder to save the model outputs. Default is the current working directory.
        --prefix (str): Job prefix to use for output files. Default is 'job'.
        --log_transform (str): Whether to apply log-transformation to input data matrices. Choices are ['True', 'False']. Default is 'False'.
        --early_stop_patience (int): How many epochs to wait when no improvements in validation loss are observed. Default is 10; set to -1 to disable early stopping.
        --hpo_patience (int): How many hyperparameter optimisation iterations to wait for when no improvements are observed. Default is 10; set to 0 to disable early stopping.
        --use_cv (bool): If set, a 5-fold cross-validation training will be done. Otherwise, a single training on 80 percent of the dataset is done.
        --use_loss_weighting (str): Whether to apply loss-balancing using uncertainty weights method. Choices are ['True', 'False']. Default is 'True'.
        --evaluate_baseline_performance (str): Whether to run Random Forest + SVMs to see the performance of off-the-shelf tools on the same dataset. Choices are ['True', 'False']. Default is 'True'.
        --threads (int): How many threads to use when using CPU. Default is 4.
        --use_gpu (bool): If set, the system will attempt to use CUDA/GPU if available.
        --disable_marker_finding (bool): If set, marker discovery after model training is disabled.
        --string_organism (int): STRING DB organism id. Default is 9606.
        --string_node_name (str): Type of node name. Choices are ["gene_name", "gene_id"]. Default is "gene_name".
    """
    parser = argparse.ArgumentParser(description="Flexynesis - Your PyTorch model training interface", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--data_path", help="(Required) Path to the folder with train/test data files", type=str, required = True)
    parser.add_argument("--model_class", help="(Required) The kind of model class to instantiate", type=str, 
                        choices=["DirectPred", "supervised_vae", "MultiTripletNetwork", "CrossModalPred", "GNN"], required = True)
    parser.add_argument("--gnn_conv_type", help="If model_class is set to GNN, choose which graph convolution type to use", type=str, 
                        choices=["GC", "GCN", "SAGE"])
    parser.add_argument("--target_variables", 
                        help="(Optional if survival variables are not set to None)." 
                        "Which variables in 'clin.csv' to use for predictions, comma-separated if multiple", 
                        type = str, default = None)
    parser.add_argument("--batch_variables", 
                        help="(Optional) Which variables in 'clin.csv' to use for data integration / batch correction, comma-separated if multiple", 
                        type = str, default = None)
    parser.add_argument("--surv_event_var", help="Which column in 'clin.csv' to use as event/status indicator for survival modeling", type = str, default = None)
    parser.add_argument("--surv_time_var", help="Which column in 'clin.csv' to use as time/duration indicator for survival modeling", type = str, default = None)
    parser.add_argument('--config_path', type=str, default=None, help='Optional path to an external hyperparameter configuration file in YAML format.')
    parser.add_argument("--fusion_type", help="How to fuse the omics layers", type=str, choices=["early", "intermediate"], default = 'intermediate')
    parser.add_argument("--hpo_iter", help="Number of iterations for hyperparameter optimisation", type=int, default = 5)
    parser.add_argument("--finetuning_samples", help="Number of samples from the test dataset to use for fine-tuning the model. Set to 0 to disable fine-tuning", type=int, default = 0)
    parser.add_argument("--variance_threshold", help="Variance threshold (as percentile) to drop low variance features (default is 1; set to 0 for no variance filtering)", type=float, default = 1)
    parser.add_argument("--correlation_threshold", help="Correlation threshold to drop highly redundant features (default is 0.8; set to 1 for no redundancy filtering)", type=float, default = 0.8)
    parser.add_argument("--restrict_to_features", help="Restrict the analyis to the list of features provided by the user (default is None)", type = str, default = None)
    parser.add_argument("--subsample", help="Downsample training set to randomly drawn N samples for training. Disabled when set to 0", type=int, default = 0)
    parser.add_argument("--features_min", help="Minimum number of features to retain after feature selection", type=int, default = 500)
    parser.add_argument("--features_top_percentile", help="Top percentile features (among the features remaining after variance filtering and data cleanup to retain after feature selection", type=float, default = 20)
    parser.add_argument("--data_types", help="(Required) Which omic data matrices to work on, comma-separated: e.g. 'gex,cnv'", type=str, required = True)
    parser.add_argument("--input_layers", 
                        help="If model_class is set to CrossModalPred, choose which data types to use as input/encoded layers"
                        "Comma-separated if multiple",
                        type=str, default = None
                        )
    parser.add_argument("--output_layers", 
                        help="If model_class is set to CrossModalPred, choose which data types to use as output/decoded layers"
                        "Comma-separated if multiple",
                        type=str, default = None
                        )    
    parser.add_argument("--outdir", help="Path to the output folder to save the model outputs", type=str, default = os.getcwd())
    parser.add_argument("--prefix", help="Job prefix to use for output files", type=str, default = 'job')
    parser.add_argument("--log_transform", help="whether to apply log-transformation to input data matrices", type=str, choices=['True', 'False'], default = 'False')
    parser.add_argument("--early_stop_patience", help="How many epochs to wait when no improvements in validation loss is observed (default 10; set to -1 to disable early stopping)", type=int, default = 10)
    parser.add_argument("--hpo_patience", help="How many hyperparamater optimisation iterations to wait for when no improvements are observed (default is 10; set to 0 to disable early stopping)", type=int, default = 10)
    parser.add_argument("--use_cv", action="store_true", 
                        help="(Optional) If set, the a 5-fold cross-validation training will be done. Otherwise, a single trainig on 80 percent of the dataset is done.")
    parser.add_argument("--use_loss_weighting", help="whether to apply loss-balancing using uncertainty weights method", type=str, choices=['True', 'False'], default = 'True')
    parser.add_argument("--evaluate_baseline_performance", help="whether to run Random Forest + SVMs to see the performance of off-the-shelf tools on the same dataset", type=str, choices=['True', 'False'], default = 'True')
    parser.add_argument("--threads", help="(Optional) How many threads to use when using CPU (default is 4)", type=int, default = 4)
    parser.add_argument("--use_gpu", action="store_true", 
                        help="(Optional) If set, the system will attempt to use CUDA/GPU if available.")
    parser.add_argument("--disable_marker_finding", action="store_true", 
                        help="(Optional) If set, marker discovery after model training is disabled.")
    # GNN args.
    parser.add_argument("--string_organism", help="STRING DB organism id.", type=int, default=9606)
    parser.add_argument("--string_node_name", help="Type of node name.", type=str, choices=["gene_name", "gene_id"], default="gene_name")

    args = parser.parse_args()
    
    # do some sanity checks on input arguments
    # 1. Check for survival variables consistency
    if (args.surv_event_var is None) != (args.surv_time_var is None):
        parser.error("Both --surv_event_var and --surv_time_var must be provided together or left as None.")

    # 2. Check for required variables for model classes
    if args.model_class != "supervised_vae" and args.model_class != 'CrossModalPred':
        if not any([args.target_variables, args.surv_event_var, args.batch_variables]):
            parser.error(''.join(["When selecting a model other than 'supervised_vae' or 'CrossModalPred',",
                                  "you must provide at least one of --target_variables, ",
                                  "survival variables (--surv_event_var and --surv_time_var)",
                                  "or --batch_variables."]))

    # 3. Check for compatibility of fusion_type with GNN
    if args.fusion_type == "early":
        if args.model_class == 'CrossModalPred': 
            parser.error("The 'CrossModalPred' model cannot be used with early fusion type. "
                         "Use --fusion_type intermediate instead.")
            
    
    # 4. Check for device availability if --accelerator is set. 
    if args.use_gpu:
        if not torch.cuda.is_available():
            warnings.warn(''.join(["\n\n!!! WARNING: GPU REQUESTED BUT NOT AVAILABLE. FALLING BACK TO CPU.\n",
                                   "PERFORMANCE MAY BE DEGRADED\n",
                                   "IF USING A SLURM SCHEDULER, ENSURE YOU REQUEST A GPU WITH: ",
                                   "`srun --gpus=1 --pty flexynesis <rest of your_command>` !!!\n\n"]))
            time.sleep(3)  #wait a bit to capture user's attention to the warning
            device_type = 'cpu'
            torch.set_num_threads(args.threads)
        else:
            device_type = 'gpu'
    else:
        device_type = 'cpu'
        torch.set_num_threads(args.threads)

    # 5. check GNN arguments
    if args.model_class == 'GNN':
        if not args.gnn_conv_type:
            warning_message = "\n".join([
                "\n\n!!! When running GNN, a convolution type can be set",
                "with the --gnn_conv_type flag. See `flexynesis -h` for full set of options.",
                "Falling back on the default convolution type: GC !!!\n\n"
            ])
            warnings.warn(warning_message)
            time.sleep(3)  #wait a bit to capture user's attention to the warning
            gnn_conv_type = 'GC'
        else:
            gnn_conv_type = args.gnn_conv_type
    else:
        gnn_conv_type = None
        
    # 6. Check CrossModalPred arguments
    input_layers = args.input_layers
    output_layers = args.output_layers
    datatypes = args.data_types.strip().split(',')
    if args.model_class == 'CrossModalPred':
        # check if input output layers are matching the requested data types 
        if args.input_layers: 
            input_layers = input_layers.strip().split(',')
            # Check if input_layers are a subset of datatypes
            if not all(layer in datatypes for layer in input_layers):
                raise ValueError(f"Input layers {input_layers} are not a valid subset of the data types: ({datatypes}).")
        # check if output_layers are a subset of datatypes
        if args.output_layers: 
            output_layers = output_layers.strip().split(',')
            if not all(layer in datatypes for layer in output_layers):
                raise ValueError(f"Output layers {output_layers} are not a valid subset of the data types: ({datatypes}).")
        
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
        CrossModalPred: tuple[CrossModalPred, str] = CrossModalPred, "CrossModalPred"
        GNN: tuple[GNN, str] = GNN, "GNN"
        
    available_models = AvailableModels()
    model_class = getattr(available_models, args.model_class, None)
    if model_class is None:
        raise ValueError(f"Invalid model_class: {args.model_class}")
    else:
        model_class, config_name = model_class

    # import assays and labels
    inputDir = args.data_path
    
    # Set concatenate to True to use early fusion, otherwise it will run intermediate fusion
    # Currently, GNNs will only work in early fusion mode, but requires the data to be not concatenated 
    concatenate = args.fusion_type == 'early' and args.model_class != 'GNN' 
        
    data_importer = flexynesis.DataImporter(path = args.data_path, 
                                            data_types = datatypes,
                                            concatenate = concatenate, 
                                            log_transform = args.log_transform == 'True',
                                            variance_threshold = args.variance_threshold/100,  
                                            correlation_threshold = args.correlation_threshold,
                                            restrict_to_features = args.restrict_to_features,
                                            min_features= args.features_min, 
                                            top_percentile= args.features_top_percentile,
                                            processed_dir = '_'.join(['processed', args.prefix]),
                                            downsample = args.subsample)
    train_dataset, test_dataset = data_importer.import_data()
    
    if args.model_class == 'GNN': 
        # overlay datasets with network info 
        # this is a temporary solution 
        print("[INFO] Overlaying the dataset with network data from STRINGDB")
        obj = STRING(os.path.join(args.data_path, '_'.join(['processed', args.prefix])), 
                     args.string_organism, args.string_node_name)
        train_dataset = MultiOmicDatasetNW(train_dataset, obj.graph_df)
        train_dataset.print_stats()
        test_dataset = MultiOmicDatasetNW(test_dataset, obj.graph_df)
        
    
    # print feature logs to file (we use these tables to track which features are dropped/selected and why)
    feature_logs = data_importer.feature_logs
    for key in feature_logs.keys():
        feature_logs[key].to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'feature_logs', key, 'csv'])), 
                                 header=True, index=False)

    # define a tuner object, which will instantiate a DirectPred class 
    # using the input dataset and the tuning configuration from the config.py
    tuner = flexynesis.HyperparameterTuning(dataset = train_dataset, 
                                            model_class = model_class, 
                                            target_variables = args.target_variables.strip().split(',') if args.target_variables is not None else [],
                                            batch_variables = args.batch_variables.strip().split(',') if args.batch_variables is not None else None,
                                            surv_event_var = args.surv_event_var,
                                            surv_time_var = args.surv_time_var,
                                            config_name = config_name, 
                                            config_path = args.config_path,
                                            n_iter=int(args.hpo_iter),
                                            use_loss_weighting = args.use_loss_weighting == 'True',
                                            use_cv = args.use_cv, 
                                            early_stop_patience = int(args.early_stop_patience), 
                                            device_type = device_type,
                                            gnn_conv_type = gnn_conv_type,
                                            input_layers = input_layers,
                                            output_layers = output_layers)    
    
    # do a hyperparameter search training multiple models and get the best_configuration 
    model, best_params = tuner.perform_tuning(hpo_patience = args.hpo_patience)
        
    # if fine-tuning is enabled; fine tune the model on a portion of test samples 
    if args.finetuning_samples > 0:
        finetuneSampleN = args.finetuning_samples
        print("[INFO] Finetuning the model on ",finetuneSampleN,"test samples")
        # split test dataset into finetuning and holdout datasets 
        all_indices = range(len(test_dataset))
        finetune_indices = random.sample(all_indices, finetuneSampleN)
        holdout_indices = list(set(all_indices) - set(finetune_indices))
        finetune_dataset = test_dataset.subset(finetune_indices)
        holdout_dataset = test_dataset.subset(holdout_indices)
        
        # fine tune on the finetuning dataset; freeze the encoders 
        finetuner = flexynesis.FineTuner(model, 
                                         finetune_dataset)
        finetuner.run_experiments()
            
        # update the model to finetuned model 
        model = finetuner.model 
        # update the test dataset to exclude finetuning samples
        test_dataset = holdout_dataset 
    
    # evaluate predictions;  (if any supervised learning happened)
    if any([args.target_variables, args.surv_event_var, args.batch_variables]):
        print("[INFO] Computing model evaluation metrics")
        metrics_df = flexynesis.evaluate_wrapper(model.predict(test_dataset), test_dataset, 
                                                 surv_event_var=model.surv_event_var, 
                                                 surv_time_var=model.surv_time_var)
        metrics_df.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'stats.csv'])), header=True, index=False)

        # print known/predicted labels 
        predicted_labels = pd.concat([flexynesis.get_predicted_labels(model.predict(train_dataset), train_dataset, 'train'),
                                      flexynesis.get_predicted_labels(model.predict(test_dataset), test_dataset, 'test')], 
                                    ignore_index=True)
        predicted_labels.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'predicted_labels.csv'])), header=True, index=False)
        
        if not args.disable_marker_finding: # unless marker discovery is disabled
            # compute feature importance values
            print("[INFO] Computing variable importance scores")
            for var in model.target_variables:
                model.compute_feature_importance(train_dataset, var, steps = 25)
            df_imp = pd.concat([model.feature_importances[x] for x in model.target_variables], 
                               ignore_index = True)
            df_imp.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'feature_importance.csv'])), header=True, index=False)

    # get sample embeddings and save 
    print("[INFO] Extracting sample embeddings")
    embeddings_train = model.transform(train_dataset)
    embeddings_test = model.transform(test_dataset)
    
    embeddings_train.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'embeddings_train.csv'])), header=True)
    embeddings_test.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'embeddings_test.csv'])), header=True)
    
    # also filter embeddings to remove batch-associated dims and only keep target-variable associated dims 
    if args.batch_variables is not None:
        print("[INFO] Printing filtered embeddings")
        embeddings_train_filtered = flexynesis.remove_batch_associated_variables(data = embeddings_train, 
                                                                                 batch_dict={x: train_dataset.ann[x] for x in model.batch_variables} if model.batch_variables is not None else None, 
                                                                                 target_dict={x: train_dataset.ann[x] for x in model.target_variables}, 
                                                                                 variable_types=train_dataset.variable_types)
        # filter test embeddings to keep the same dims as the filtered training embeddings
        embeddings_test_filtered = embeddings_test[embeddings_train_filtered.columns]

        # save 
        embeddings_train_filtered.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'embeddings_train.filtered.csv'])), header=True)    
        embeddings_test_filtered.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'embeddings_test.filtered.csv'])), header=True)    

    # for architectures with decoders; print decoded output layers 
    if args.model_class == 'CrossModalPred':
        print("[INFO] Printing decoded output layers")
        output_layers_train = model.decode(train_dataset)
        output_layers_test = model.decode(test_dataset)
        for layer in output_layers_train.keys():
            output_layers_train[layer].to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'train_decoded', layer, 'csv'])), header=True)
        for layer in output_layers_test.keys():
            output_layers_test[layer].to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'test_decoded', layer, 'csv'])), header=True)

    
    # evaluate off-the-shelf methods on the main target variable 
    if args.evaluate_baseline_performance == 'True':
        print("[INFO] Computing off-the-shelf method performance on first target variable:",model.target_variables[0])
        var = model.target_variables[0]
        metrics = pd.DataFrame()
        
        # in the case when GNNEarly was used, the we use the initial multiomicdataset for train/test
        # because GNNEarly requires a modified dataset structure to fit the networks (temporary solution)
        train = train_dataset.multiomic_dataset if args.model_class == 'GNN' else train_dataset
        test = test_dataset.multiomic_dataset if args.model_class == 'GNN' else test_dataset
        
        if var != model.surv_event_var: 
            metrics = flexynesis.evaluate_baseline_performance(train, test, 
                                                            variable_name = var, 
                                                            n_folds=5,
                                                            n_jobs = int(args.threads))
        if model.surv_event_var and model.surv_time_var:
            print("[INFO] Computing off-the-shelf method performance on survival variable:",model.surv_time_var)
            metrics_baseline_survival = flexynesis.evaluate_baseline_survival_performance(train, test, 
                                                                                             model.surv_time_var, 
                                                                                             model.surv_event_var, 
                                                                                             n_folds = 5,
                                                                                             n_jobs = int(args.threads))
            metrics = pd.concat([metrics, metrics_baseline_survival], axis = 0, ignore_index = True)
        
        if not metrics.empty:
            metrics.to_csv(os.path.join(args.outdir, '.'.join([args.prefix, 'baseline.stats.csv'])), header=True, index=False) 
    
    # save the trained model in file
    torch.save(model, os.path.join(args.outdir, '.'.join([args.prefix, 'final_model.pth'])))
    
if __name__ == "__main__":
    main()
