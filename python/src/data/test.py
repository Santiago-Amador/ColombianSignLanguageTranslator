from data_loader import DataLoader

loader = DataLoader(img_size=(128, 128))
X_train, X_test, y_train, y_test = loader.load_dataset_split()
