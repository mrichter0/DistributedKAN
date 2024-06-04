# DistributedKAN. Distributed training on multi-gpus using FastKAN model for large datasets
For binary classification on large datasets, FastKAN (https://github.com/ZiyaoLi/fast-kan) produced better AUPR values than CatBoost. However, FastKAN had no options to utilize multiple GPUs. DistributedDataParallel and DataParallel were not compatible with my cluster, so I built a server capable of micromanaging multiple GPUs and handling parameter passing through a ParameterServer. I compared my server to running a single instance of FastKAN and it produced similar AUPR results, but ran 3-4 times faster (utilizing 4 GPU). I now use DistributedKAN with 10 GPUs on the same node for training of 100 million samples with 300 gb of features and the training can be completed in about an hour.

If you are using a notebook load your data in the cell above, import and call main with your data:
import importlib
import DistributedKAN
importlib.reload(DistributedKAN) #recommend using this to reload every time 
from DistributedKAN import main
main(train_loader, test_loaders)

Or to run from python, I add my data loading in the same script, then run DistributedKAN (shown in load_data.py)

def load_data():
...
    return train_loader, test_loader
    
if __name__ == '__main__':
    import DistributedKAN
    from DistributedKAN import main
    
    train_loader, test_loader = load_data()
    main(train_loader, test_loader)


DistributedKAN.py - micromanage multiple GPUs and handle parameter passing through a ParameterServer \
from models import FastKAN  # Import the FastKAN model

class ParameterServer(nn.Module) Here Parameters are passed from each GPU and are stored and averaged. This allows learning to occur simultaneously between each GPU.
def server_process(model, parameter_queues) Handles passing of parameters very effeciently
def calculate_aupr(model, optimizer_state_dict, test_loader, rank, epoch, avg_epoch_loss, epoch_start_time) \ This allows for asynchronous evaluation of the model, which increases the speed of training. Calculates aupr, and the model is saved if criteria is met. 
def worker_process(rank, model, train_loader, test_loader, parameter_queue, device):  The worker_process handles the training, calls calculate_aupr for evaluation, sends/receives parameters from the ParameterServer.
def find_and_load_best_model: searches for and loads the best previously saved model in the directory.
main: Starts all processes. Splits the training data into n groups based on the number of available GPUs. 

See advanced notes for additional tips and troubleshooting!
