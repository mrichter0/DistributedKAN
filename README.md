#### Implementation Details

### DistributedKAN.py - Micromanage multiple GPUs and handle parameter passing through a ParameterServer

```python
# Import the FastKAN model
from models import FastKAN  
```

```python
# ParameterServer(nn.Module)
# Here parameters are passed from each GPU and are stored and averaged. 
# This allows learning to occur simultaneously between each GPU.
class ParameterServer(nn.Module):
    ...
```

```python
# server_process(model, parameter_queues)
# Handles passing of parameters very efficiently.
def server_process(model, parameter_queues):
    ...
```

```python
# calculate_aupr(model, optimizer_state_dict, test_loader, rank, epoch, avg_epoch_loss, epoch_start_time)
# This allows for asynchronous evaluation of the model, which increases the speed of training.
# It calculates AUPR, and the model is saved if criteria are met.
def calculate_aupr(model, optimizer_state_dict, test_loader, rank, epoch, avg_epoch_loss, epoch_start_time):
    ...
```

```python
# worker_process(rank, model, train_loader, test_loader, parameter_queue, device)
# The worker process handles the training, calls `calculate_aupr` for evaluation,
# and sends/receives parameters from the ParameterServer.
def worker_process(rank, model, train_loader, test_loader, parameter_queue, device):
    ...
```

```python
# find_and_load_best_model
# Searches for and loads the best previously saved model in the directory.
def find_and_load_best_model():
    ...
```

```python
# main
# Starts all processes and splits the training data into groups based on the number of available GPUs.
def main():
    ...
```
