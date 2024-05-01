import yaml
label_dict = {}
with open('labels.yml', 'r') as f:
    label_dict = yaml.safe_load(f)

print(label_dict)
print('Animesh' in label_dict)