import os
import shutil
import kagglehub

# Download the dataset (downloads to KaggleHub cache)
path = kagglehub.dataset_download("hojjatk/mnist-dataset")

# Define your target folder
target_dir = os.path.join(os.getcwd(), "data")

# Create target folder if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Copy dataset files to ./data/
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Dataset copied to:", target_dir)
