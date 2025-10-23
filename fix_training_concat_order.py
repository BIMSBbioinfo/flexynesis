with open('flexynesis/data.py', 'r') as f:
    content = f.read()

# Fix training concatenation to use sorted order
old_training = """        if self.concatenate:
            training_dataset.dat = {'all': torch.cat([training_dataset.dat[x] for x in training_dataset.dat.keys()], dim = 1)}
            training_dataset.features = {'all': list(chain(*training_dataset.features.values()))}
        
            testing_dataset.dat = {'all': torch.cat([testing_dataset.dat[x] for x in testing_dataset.dat.keys()], dim = 1)}
            testing_dataset.features = {'all': list(chain(*testing_dataset.features.values()))}"""

new_training = """        if self.concatenate:
            # Use data_types order for consistent concatenation
            modality_order = self.data_types
            training_dataset.dat = {'all': torch.cat([training_dataset.dat[x] for x in modality_order], dim = 1)}
            training_dataset.features = {'all': list(chain(*[training_dataset.features[x] for x in modality_order]))}
        
            testing_dataset.dat = {'all': torch.cat([testing_dataset.dat[x] for x in modality_order], dim = 1)}
            testing_dataset.features = {'all': list(chain(*[testing_dataset.features[x] for x in modality_order]))}"""

content = content.replace(old_training, new_training)

with open('flexynesis/data.py', 'w') as f:
    f.write(content)

print("Fixed: Early fusion now uses data_types order consistently")
