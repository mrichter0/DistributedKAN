# DistributedKAN. Distributed training on multi-gpus using FastKAN model for large datasets
#For binary classification on large datasets, FastKAN (https://github.com/ZiyaoLi/fast-kan) produced better AUPR values than CatBoost for smaller subsets. However, when upscaling, FastKAN had poor performance with long epoch times, very little GPU usage, and no options to utilize multiple GPUs. DistributedDataParallel and DataParallel were not compatible with my cluster, so I built a server capable of micromanaging multiple GPUs and handling parameter passing through a ParameterServer. I compared my server to running a single instance of FastKAN and it produced similar AUPR results, but ran 3-4 times faster (utilizing 4 GPU). DistributedKAN has been tested using up to 10 GPUs on the same node.
#
#   
#  
# 
#
# 
