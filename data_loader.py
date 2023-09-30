import torch.utils.data as data_utils
import torch
import os
import shutil

def get_loader(features, batch_size, num_workers=1):
    """
    Build and return a data loader.
    """
    dataset = data_utils.TensorDataset(torch.Tensor(features))
    loader = data_utils.DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle = False,
                        num_workers=num_workers
                        )
    return loader



def create_dirs_if_not_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_list):
            os.makedirs(dir_list)

def delete_dirs_if_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if os.path.exists(dir):
                shutil.rmtree(dir)
    else:
        if os.path.exists(dir_list):
            shutil.rmtree(dir_list)
