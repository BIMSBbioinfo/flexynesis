with open('flexynesis/data.py', 'r') as f:
    content = f.read()

# After creating the dataset, add GNN conversion logic
old_return = """        # Concatenate for early fusion if needed
        if self.modalities == ['all']:
            from itertools import chain
            # Concatenate all modalities
            dataset.dat = {'all': torch.cat([dataset.dat[x] for x in dataset.dat.keys()], dim=1)}
            all_features = list(chain(*dataset.features.values()))
            dataset.features = {'all': all_features}
            
            # Filter to expected features from artifacts
            expected_all_features = self.feature_names['all']
            feature_indices = [i for i, f in enumerate(all_features) if f in expected_all_features]
            dataset.dat['all'] = dataset.dat['all'][:, feature_indices]
            dataset.features['all'] = [all_features[i] for i in feature_indices]
        
        return dataset"""

new_return = """        # Concatenate for early fusion if needed
        if self.modalities == ['all']:
            from itertools import chain
            # Concatenate all modalities
            dataset.dat = {'all': torch.cat([dataset.dat[x] for x in dataset.dat.keys()], dim=1)}
            all_features = list(chain(*dataset.features.values()))
            dataset.features = {'all': all_features}
            
            # Filter to expected features from artifacts
            expected_all_features = self.feature_names['all']
            feature_indices = [i for i, f in enumerate(all_features) if f in expected_all_features]
            dataset.dat['all'] = dataset.dat['all'][:, feature_indices]
            dataset.features['all'] = [all_features[i] for i in feature_indices]
        
        return dataset
    
    def convert_to_gnn_dataset(self, dataset, feature_ann_path):
        '''Convert MultiOmicDataset to MultiOmicDatasetNW for GNN models'''
        from .data import MultiOmicDatasetNW
        import pandas as pd
        
        # Load feature annotations if provided
        feature_ann = None
        if feature_ann_path and os.path.exists(feature_ann_path):
            feature_ann = pd.read_csv(feature_ann_path, index_col=0)
        
        return MultiOmicDatasetNW(dataset, feature_ann)"""

content = content.replace(old_return, new_return)

with open('flexynesis/data.py', 'w') as f:
    f.write(content)

print("Added GNN dataset conversion method")
