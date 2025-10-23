with open('flexynesis/__main__.py', 'r') as f:
    content = f.read()

old_line = "                'data_types': args.data_types.split(','),  # Use CLI order"
new_line = "                'data_types': list(data_importer.train_features.keys()) if hasattr(data_importer, 'train_features') else args.data_types.split(','),  # Use actual data structure keys (e.g. ['all'] for early fusion)"

content = content.replace(old_line, new_line)

with open('flexynesis/__main__.py', 'w') as f:
    f.write(content)

print("Fixed: data_types now uses train_features.keys() to handle early fusion correctly")
