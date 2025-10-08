import kagglehub

# Download latest version
path = kagglehub.dataset_download("alexjhon82/under-water-sensor-dataset")

print("Path to dataset files:", path)