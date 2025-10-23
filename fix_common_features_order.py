with open('flexynesis/data.py', 'r') as f:
    content = f.read()

old_return = "        return list(all_omic_features.intersection(interaction_genes))"
new_return = "        return sorted(list(all_omic_features.intersection(interaction_genes)))"

content = content.replace(old_return, new_return)

with open('flexynesis/data.py', 'w') as f:
    f.write(content)

print("Fixed: common_features now sorted for consistent order")
