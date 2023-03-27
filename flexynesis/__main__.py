import argparse
import os
import yaml
import torch
import pandas as pd
import flexynesis

def filter(dat, minFeatures):
    counts = {x: max(int(dat[x].shape[0]/20), minFeatures) for x in dat.keys()}
    dat = {x: flexynesis.filter_by_laplacian(dat[x].T, topN=counts[x]).T for x in dat.keys()}
    return dat

# subset dat2 to only include features available in dat1
def harmonize(dat1, dat2):
    # features to keep
    features = {x: dat1[x].index for x in dat1.keys()}
    return {x: dat2[x].loc[features[x]] for x in dat2.keys()}

def main():
    parser = argparse.ArgumentParser(description="Flexynesis - Your PyTorch model training interface")
    
    parser.add_argument("data_path", help="Path to the folder with train/test data files", type=str)
    parser.add_argument("model_class", help="The kind of model class to instantiate", type=str, choices=["DirectPred", "supervised_vae", "MultiTripletNetwork"])
    parser.add_argument("config_file", help="Path to the config.yaml file", type=str)
    # import data TODO: make these arguments as well
    datatypes = ['layer1', 'layer2']
    drugName = 'Erlotinib'
    torch.set_num_threads(4)

    args = parser.parse_args()

    # Load the config file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

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
    dat_train = {x: pd.read_csv(os.path.join(inputDir, 'train', ''.join([x, '.csv']))) for x in datatypes}
    dat_test = {x: pd.read_csv(os.path.join(inputDir, 'test', ''.join([x, '.holdout.csv']))) for x in datatypes}

    # get drug response data (concatenate to keep all labels in one df)
    drugs = pd.concat([pd.read_csv(os.path.join(inputDir, 'train', 'clin.csv'), sep = '\t').transpose(),
                      pd.read_csv(os.path.join(inputDir, 'test', 'clin.csv'), sep = '\t').transpose()])

    # feature selection
    dat_train = filter(dat_train, int(config['feature_selection']['min']))
    dat_test = harmonize(dat_train, dat_test)

    # Set concatenate to True to use early fusion, otherwise it will run intermediate fusion
    concatenate = False
    if config['fusion'] == 'early':
        concatenate = True
    train_dataset = flexynesis.data.make_dataset(dat_train, drugs, drugName, concatenate = concatenate)
    test_dataset = flexynesis.data.make_dataset(dat_test, drugs, drugName, concatenate = concatenate)
    
    # define a tuner object, which will instantiate a DirectPred class 
    # using the input dataset and the tuning configuration from the config.py
    tuner = flexynesis.HyperparameterTuning(train_dataset, model_class = model_class, 
                                            config_name = config_name, 
                                            n_iter=int(config['hyperparameter_tuning']['n_iter']))    
    
    # do a hyperparameter search training multiple models and get the best_configuration 
    model, best_params = tuner.perform_tuning()
    
    # evaluate the model on test dataset
    COR = model.evaluate(test_dataset)
    stats = pd.DataFrame.from_dict({'RMSE': 'NA', 'Rsquare': 'NA', 'COR': COR, 
                                        'drug': drugName, 'trainSampleN': len(train_dataset), 
                                        'testSampleN': len(test_dataset), 
                                        'tool': args.model_class}, orient = 'index').T
    
    # save stats 
    outFile = os.path.join('.'.join(['stats', drugName, 'tsv']))
    print("Saving stats to file", outFile)
    stats.to_csv(outFile, index = False, sep = '\t')

if __name__ == "__main__":
    main()
