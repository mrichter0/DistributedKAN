# DistributedKAN. Distributed training on multi-gpus using FastKAN model for large datasets
For binary classification on large datasets, FastKAN (https://github.com/ZiyaoLi/fast-kan) produced better AUPR values than CatBoost for smaller subsets. However, when upscaling, FastKAN had poor performance with long epoch times, very little GPU usage, and no options to utilize multiple GPUs. DistributedDataParallel and DataParallel were not compatible with my cluster, so I built a server capable of micromanaging multiple GPUs and handling parameter passing through a ParameterServer. I compared my server to running a single instance of FastKAN and it produced similar AUPR results, but ran 3-4 times faster (utilizing 4 GPU). DistributedKAN has been tested using up to 10 GPUs on the same node.

load_data.py - currently setup to use TensorDataset and DataLoader \
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)  \
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=thread_num)  \
    from DistributedKAN import main  \
    main(train_loader, test_loader \

    
DistributedKAN.py - micromanage multiple GPUs and handle parameter passing through a ParameterServer \
models.py - FastKAN model as presented by https://github.com/ZiyaoLi/fast-kan \
    train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_labels_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=thread_num)
    


DistributedKAN.py - micromanage multiple GPUs and handle parameter passing through a ParameterServer
    def main(train_loader, test_loader) 

#  
# 
#
# 
