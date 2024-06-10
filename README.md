# DistributedKAN: Distributed Training on Multi-GPUs Using FastKAN Model for Large Datasets

## Overview
DistributedKAN leverages the FastKAN model for binary classification on large datasets, achieving superior AUPR values compared to CatBoost. Unlike FastKAN, which does not support multiple GPUs, DistributedKAN utilizes a custom server to manage multiple GPUs and handle parameter passing through a ParameterServer.

Using DistributedKAN, I achieved similar AUPR results as a single instance of FastKAN but with a 3-4x speed increase by utilizing 4 GPUs. Currently, DistributedKAN is used with 10 GPUs on the same node to train 100 million samples with 300 GB of features in about an hour.

## Usage

### In a Jupyter Notebook
Load your data in the cell above, then import and call `main` with your data:

```python
import importlib
import DistributedKAN
importlib.reload(DistributedKAN)  # Recommended to reload every time
from DistributedKAN import main

# Replace `train_loader` and `test_loader` with your data loaders
main(train_loader, test_loader)
```

#### Implementation Details

### DistributedKAN.py - Micromanage multiple GPUs and handle parameter passing through a ParameterServer

```python
# Import the FastKAN model. FastKAN is placed into models.py, with no modifications
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
### Advanced Notes
See advanced notes for additional tips and troubleshooting.

### References
FastKAN: https://github.com/ZiyaoLi/fast-kan
