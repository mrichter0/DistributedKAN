import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc

def load_data():
    print('Loading data...', flush=True)
    dataset_path = 'path/to/your/dataset/'
    train_labels_path = 'path/to/your/train_labels.csv'
    test_labels_path = 'path/to/your/test_labels.csv'
    thread_num = 64
    os.environ['OMP_NUM_THREADS'] = '64'
    batch_size = 512

    # Load training data
    train_data_file = os.path.join(dataset_path, 'train_data.pkl')
    with open(train_data_file, 'rb') as file:
        train_data = pickle.load(file)
    train_labels = pd.read_csv(train_labels_path)['label'].values
    print(f'train_data shape: {train_data.shape}')
    print(f'train_labels shape: {train_labels.shape}')
    
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=thread_num)
    
    del train_data, train_data_tensor, train_labels_tensor
    gc.collect()
    
    # Load test data
    test_data_file = os.path.join(dataset_path, 'test_data.pkl')
    with open(test_data_file, 'rb') as file:
        test_data = pickle.load(file)
    test_labels = pd.read_csv(test_labels_path)['label'].values

    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_labels_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=thread_num)

    del test_data, test_data_tensor, test_labels_tensor
    gc.collect()

    print('Data loaded. Starting model...', flush=True)

    return train_loader, test_loader

if __name__ == '__main__':
    import DistributedKAN
    from DistributedKAN import main

    train_loader, test_loader = load_data()
    main(train_loader, test_loader)
