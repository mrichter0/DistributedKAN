# DistributedKAN. Distributed training on multi-gpus using FastKAN model for large datasets
For binary classification on large datasets, FastKAN (https://github.com/ZiyaoLi/fast-kan) produced better AUPR values than CatBoost. However, FastKAN had no options to utilize multiple GPUs. DistributedDataParallel and DataParallel were not compatible with my cluster, so I built a server capable of micromanaging multiple GPUs and handling parameter passing through a ParameterServer. I compared my server to running a single instance of FastKAN and it produced similar AUPR results, but ran 3-4 times faster (utilizing 4 GPU). I now use DistributedKAN with 10 GPUs on the same node for training of 100 million samples with 300 gb of features and the training can be completed in under and hour.

models.py - FastKAN model as presented by https://github.com/ZiyaoLi/fast-kan \
load_data.py - currently setup to use TensorDataset and DataLoader \ imports main from DistributedKAN
DistributedKAN.py - micromanage multiple GPUs and handle parameter passing through a ParameterServer \

class ParameterServer(nn.Module) Here Parameters are passed from each GPU and are stored and averaged. This allows learning to occur simultaneously between each GPU.
def server_process(model, parameter_queues) Handles passing of parameters very effeciently
def calculate_aupr(model, optimizer_state_dict, test_loader, rank, epoch, avg_epoch_loss, epoch_start_time
