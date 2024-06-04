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
