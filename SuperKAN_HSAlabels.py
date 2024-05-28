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
    dataset_path = '/data/home/mrichte3/DTIAM/code/5-20-24/'
    train_labels_path = '../submission/HSA/HSA_train.csv'
    test_labels_path = '../submission/HSA/HSA_test.csv'
    thread_num = 64
    os.environ['OMP_NUM_THREADS'] = '64'
    batch_size = 512

    # comp_features_train = []
    # training_length = 10000000
    # test_length = training_length // 10
    # for i in range(17):
    #     file_name = os.path.join(dataset_path, f'train_fold_smiles.csvcomp_feat_{i}.pkl')
    #     with open(file_name, 'rb') as file:
    #         comp_feat = pickle.load(file)
    #         comp_features_train.extend(comp_feat)
    #     print(f"File {i}: {file_name}, Number of entries: {len(comp_feat)}")
    #     if len(comp_features_train) >= training_length:
    #         break
    #     gc.collect()

    # comp_features_train = comp_features_train[:training_length]
    # train_data = np.array(comp_features_train, dtype=np.float32)
    # train_labels = pd.read_csv(train_labels_path)['label'].values[:training_length]
    # pos_weight = len(train_labels) / (train_labels == 1).sum()
    # print(pos_weight)
    
    # train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    # train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    # train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=thread_num)
    
    # del comp_features_train, train_data, train_data_tensor, train_labels_tensor
    # gc.collect()
    
    # file_name = os.path.join(dataset_path, 'test_fold_smiles.csvcomp_feat_0.pkl')
    # with open(file_name, 'rb') as file:
    #     comp_features_test = pickle.load(file)
    # test_data = np.array(comp_features_test, dtype=np.float32)[:test_length]
    # test_labels = pd.read_csv(test_labels_path)['label'].values[:test_length]
    comp_features_train = []
    for i in range(17):
        file_name = os.path.join(dataset_path, f'train_fold_smiles.csvcomp_feat_{i}.pkl')
        with open(file_name, 'rb') as file:
            comp_feat = pickle.load(file)
            comp_features_train.extend(comp_feat)
        print(f"File {i}: {file_name}, Number of entries: {len(comp_feat)}")
        gc.collect()

    train_data = np.array(comp_features_train, dtype=np.float32)
    train_labels = pd.read_csv(train_labels_path)['label'].values
    print(f'train_data shape: {train_data.shape}')
    print(f'train_labels shape: {train_labels.shape}')
    pos_weight = len(train_labels) / (train_labels == 1).sum()
    print(pos_weight)
    
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=thread_num)
    
    del comp_features_train, train_data, train_data_tensor, train_labels_tensor
    gc.collect()
    
    # Load full test data
    file_name = os.path.join(dataset_path, 'test_fold_smiles.csvcomp_feat_0.pkl')
    with open(file_name, 'rb') as file:
        comp_features_test = pickle.load(file)
    test_data = np.array(comp_features_test, dtype=np.float32)
    test_labels = pd.read_csv(test_labels_path)['label'].values

    # Debug statements to check the shapes
    print(f'test_data shape: {test_data.shape}')
    print(f'test_labels shape: {test_labels.shape}')

    if test_data.shape[0] != test_labels.shape[0]:
        raise ValueError(f'Size mismatch between test_data and test_labels: {test_data.shape[0]} != {test_labels.shape[0]}')

    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    # Debug statements to check the shapes
    print(f'test_data shape: {test_data.shape}')
    print(f'test_labels shape: {test_labels.shape}')

    if test_data.shape[0] != test_labels.shape[0]:
        raise ValueError(f'Size mismatch between test_data and test_labels: {test_data.shape[0]} != {test_labels.shape[0]}')

    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    test_dataset = torch.utils.data.TensorDataset(test_data_tensor, test_labels_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=thread_num)

    del comp_features_test, test_data, test_data_tensor, test_labels_tensor
    gc.collect()

    print('Data loaded. Starting model...', flush=True)

    return train_loader, test_loader, pos_weight


if __name__ == '__main__':
    import distributed_training13_HSAlabels  # Import the module first
    from distributed_training13_HSAlabels import main


    train_loader, test_loader, pos_weight = load_data()
    main(train_loader, test_loader, pos_weight)

