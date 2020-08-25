# utils.py includes several helper functions for
# (1) dataloading
#       get_loader : Get a PyTorch loader from a PyTorch dataset
#       get_subset_loader : Get a Pytorch loader from subset of a dataset
#       get_subset_dataloaders : Get train and val loaders from train/val sets of a dataset
# (2) target_tranform functions: Map/Unmap class labels to softmax indices for training purpose
#       get_target_mapping_func: Function map real class label to softmax index
#       get_target_mapping_func_for_tensor: Function map real class label (tensor) to softmax index (tensor)
#       get_target_unmapping_dict: Dictionary map softmax index to real class label
#       get_target_unmapping_func_for_list: Function map softmax index (list) to class label (list)
# (3) training: Print mode, make dirs
#       SetPrintMode
#       makedirs
import os
import sys
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
import numpy as np

import global_setting # include some constant value

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SetPrintMode:
    def __init__(self, hidden=False):
        self.hidden = hidden

    def __enter__(self):
        if self.hidden:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hidden:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, mode=0o777)
        os.chmod(dir_name, 0o0777)

def get_loader(dataset, target_transform, batch_size=1, shuffle=False, workers=0):
    """Returns a PyTorch Dataloader
        Args:
            target_transform (fun int -> int) : None if no need to change the output label.
                                                Otherwise supply the label transform function.
            The rest of the args follows spec in torch.utils.data.Dataloader
    """
    dataset.target_transform = target_transform
    loader = DataLoader(
                 dataset,
                 batch_size=batch_size,
                 shuffle=shuffle,
                 num_workers=workers,
                 pin_memory=True,
             )
    return loader

def get_subset_loader(dataset, samples, target_transform, batch_size=1, shuffle=False, workers=0):
    subset = torch.utils.data.Subset(dataset, samples)
    subset.dataset.target_transform = target_transform
    loader = DataLoader(
                 subset,
                 batch_size=batch_size,
                 shuffle=shuffle,
                 num_workers=workers,
                 pin_memory=True
             )
    return loader

def get_subset_dataloaders(dataset, train_samples, val_samples, target_transform, batch_size, workers=0, shuffle=True):
    """ Return a dict of train/val dataloaders.
            Returns:
                dataloaders (dict) : dataloaders['train'] and dataloaders['val'] are the trainset/valset loader
            Args:
                dataset : The PyTorch dataset that contains both train and val set
                train_samples (lst of int) : The indices of train set samples
                val_samples (lst of int) : The indices of val set samples
                target_transform (fun int -> int) : Label transform function (transform the actual label to softmax index)
                other args follow specs of PyTorch
    """
    dataloaders = {}
    if len(train_samples) > 0:
        dataloaders['train'] = get_subset_loader(dataset,
                                                 train_samples,
                                                 target_transform,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 workers=workers)

    if len(val_samples) > 0:
        dataloaders['val'] = get_subset_loader(dataset,
                                               val_samples,
                                               target_transform,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               workers=workers)

    return dataloaders

def get_dataloaders(dataset, batch_size=128, workers=4, shuffle=True):
    """Return a dict of only train dataloader.
            Returns:
                dataloaders (dict) : dataloaders['train'] is the trainset loader
    """
    dataloaders = {}
    dataloaders['train'] = DataLoader(
                 dataset,
                 batch_size=batch_size,
                 shuffle=shuffle,
                 num_workers=workers,
                 pin_memory=True
             )
    return dataloaders

def get_target_mapping_func(classes,
                            discovered_classes,
                            open_classes,
                            OPEN_CLASS_INDEX=global_setting.OPEN_CLASS_INDEX,
                            UNDISCOVERED_CLASS_INDEX=global_setting.UNDISCOVERED_CLASS_INDEX):
    """Return a function that map discovered_classes indices to 0-len(discovered_classes). 
       If not in hold-out open classes, undiscovered classes
       are mapped to UNDISCOVERED_CLASS_INDEX.
       Hold-out open classes are mapped to OPEN_CLASS_INDEX.
       Always return the same indices as long as discovered classes is the same.
        Args:
            classes (lst/set of int): The list/set of all classes
            discovered_classes (set of int): The set of all discovered classes
            open_classes (set of int): The set of all hold-out open classes
            OPEN_CLASS_INDEX (int)
            UNDISCOVERED_CLASS_INDEX (int)
        Returns:
            target_mapping_func (fun int -> int) : As described above.
    """
    discovered_classes = sorted(list(discovered_classes))
    open_classes = sorted(list(open_classes))
    mapping = {idx : OPEN_CLASS_INDEX if idx in open_classes else 
                     UNDISCOVERED_CLASS_INDEX if idx not in discovered_classes else 
                     discovered_classes.index(idx)
               for idx in classes}
    return lambda idx : mapping[idx]

def get_target_mapping_func_for_tensor(classes,
                                       discovered_classes,
                                       open_classes,
                                       OPEN_CLASS_INDEX=global_setting.OPEN_CLASS_INDEX,
                                       UNDISCOVERED_CLASS_INDEX=global_setting.UNDISCOVERED_CLASS_INDEX,
                                       device='cuda'):
    """Exactly the same as get_target_mapping_func but the returning function operates on tensor level
        Returns:
            target_mapping_func (fun tensor -> tensor) : As specified.
    """
    discovered_classes = sorted(list(discovered_classes))
    open_classes = sorted(list(open_classes))
    mapping = {idx : global_setting.OPEN_CLASS_INDEX if idx in open_classes else 
                     global_setting.UNDISCOVERED_CLASS_INDEX if idx not in discovered_classes else 
                     discovered_classes.index(idx)
               for idx in classes}
    index_tensor = torch.zeros((len(classes))).long().to(device)
    for idx in classes:
        index_tensor[idx] = mapping[idx]
    def mapp_func(real_labels):
        return index_tensor[real_labels]
    return mapp_func

def get_target_unmapping_dict(classes, discovered_classes):
    """Return a dictionary that map 0-len(discovered_classes) to true discovered_classes indices.
        Args:
            classes: The list of all classes
            discovered_classes: The set of all discovered classes
        Returns:
            unmapping (dict int -> int) : It maps softmax indices to true class indices.
                                          It is the inverse function of get_target_transform_func() (for discovered_classes).
    """
    discovered_classes = sorted(list(discovered_classes))
    mapping = {idx : -1 if idx not in discovered_classes else
                     discovered_classes.index(idx)
               for idx in classes}
    unmapping = {mapping[true_index] : true_index for true_index in mapping.keys()}
    if -1 in unmapping.keys():
        del unmapping[-1]
    return unmapping

def get_target_unmapping_func_for_list(classes, discovered_classes):
    """Sames as get_target_unmapping_dict, but the returned function operate on lists.
        Returns:
            unmapping (fun lst of int -> lst of int)
    """
    discovered_classes = sorted(list(discovered_classes))
    mapping = {idx : -1 if idx not in discovered_classes else
                     discovered_classes.index(idx)
               for idx in classes}
    unmapping_dict = {mapping[true_index] : true_index for true_index in mapping.keys()}
    unmapping_dict[-1] = -1
    def unmapp_func(lst):
        return list(map(lambda x: unmapping_dict[x], lst))
    return unmapp_func


def get_dataset_info_path(save_path, data, data_config, data_rand_seed):
    dataset_info_dir = os.path.join(
                            save_path,
                            data,
                            data_config,
                        )
    if not os.path.exists(dataset_info_dir):
        print(f"{dataset_info_dir} does not exists.")
        makedirs(dataset_info_dir)
    dataset_info_path = os.path.join(dataset_info_dir, f"seed_{data_rand_seed}.pt")
    return dataset_info_path

def get_open_active_save_dir(open_active_save_dir, data, data_config, data_rand_seed, training_method, train_mode, makedir=True):
    save_dir = os.path.join(open_active_save_dir,
                            data,
                            data_config,
                            "seed_"+str(data_rand_seed),
                            "_".join([training_method, train_mode]))
    if not os.path.exists(save_dir) and makedir:
        print("Making a new directory to save checkpoints after train/query/finetune step.")
        print(f"Location {save_dir}")
        makedirs(save_dir)
    return save_dir

def get_open_save_dir(open_save_dir,
                          data,
                          data_config,
                          data_rand_seed,
                          training_method,
                          train_mode,
                          makedir=True):
    save_dir = os.path.join(open_save_dir,
                            data,
                            data_config,
                            "seed_"+str(data_rand_seed),
                            training_method,
                            train_mode)
    if not os.path.exists(save_dir) and makedir:
        print("Making a new directory to save checkpoints for open set learning.")
        print(f"Location {save_dir}")
        makedirs(save_dir)
    return save_dir

def get_active_save_dir(active_save_dir,
                        data,
                        data_config,
                        data_rand_seed,
                        training_method,
                        train_mode,
                        active_query_scheme,
                        # active_val_mode,
                        makedir=True):
    # if active_val_mode == None:
    #     training_method_str = training_method
    # else:
    #     training_method_str = "_".join([training_method, "val", active_val_mode])
    save_dir = os.path.join(active_save_dir,
                            data,
                            data_config,
                            "seed_"+str(data_rand_seed),
                            # training_method_str,
                            training_method,
                            train_mode,
                            active_query_scheme)
    if not os.path.exists(save_dir) and makedir:
        print("Making a new directory to save checkpoints for active learning.")
        print(f"Location {save_dir}")
        makedirs(save_dir)
    return save_dir

def get_trainset_info_path(data_save_path, data):
    trainset_info_dir = os.path.join(data_save_path, data)
    makedirs(trainset_info_dir)
    return os.path.join(trainset_info_dir, "trainset_info.pt")

def prepare_save_dir_from_config(config, makedir=True):
    open_set_methods = global_setting.OPEN_SET_METHOD_DICT[config.training_method]
    return prepare_save_dir(config.data_save_path,
                            config.data_download_path,
                            config.open_active_save_dir,
                            config.data,
                            config.data_config,
                            config.data_rand_seed,
                            config.training_method,
                            config.train_mode,
                            config.query_method,
                            config.budget,
                            open_set_methods,
                            makedir=makedir)


def prepare_save_dir(save_path,
                     download_path,
                     open_active_save_dir,
                     data,
                     data_config,
                     data_rand_seed,
                     training_method,
                     train_mode,
                     query_method,
                     budget,
                     open_set_methods,
                     makedir=True):
    """Return a dictionary of save_paths
    """
    paths_dict = {}
    paths_dict['data_download_path'] = download_path
    paths_dict['data_save_path'] = get_dataset_info_path(save_path,
                                                            data,
                                                            data_config,
                                                            data_rand_seed)
    paths_dict['trainset_info_path'] = get_trainset_info_path(save_path, data)
    
    # Where the training/testing results will be saved
    paths_dict['open_active_save_dir'] = get_open_active_save_dir(open_active_save_dir,
                                                          data,
                                                          data_config,
                                                          data_rand_seed,
                                                          training_method,
                                                          train_mode,
                                                          makedir=makedir)
    paths_dict['query_dir'] = os.path.join(paths_dict['open_active_save_dir'],
                                           "_".join(["active", query_method]))
    paths_dict['finetuned_dir'] = os.path.join(paths_dict['query_dir'],
                                               "_".join(["budget", str(budget)]))
    paths_dict['test_dirs'] = {}
    for open_set_method in open_set_methods:
        paths_dict['test_dirs'][open_set_method] = os.path.join(paths_dict['finetuned_dir'],
                                                                "_".join(["openset", open_set_method]))

    for folder in ["open_active_save_dir", "finetuned_dir"]:
        folder_path = paths_dict[folder]
        if not os.path.exists(folder_path) and makedir:
            print(f"Make a new folder at: {folder_path}")
            makedirs(folder_path)

    for key in paths_dict['test_dirs']:
        folder_path = paths_dict['test_dirs'][key]
        if not os.path.exists(folder_path) and makedir:
            print(f"Make a new folder at: {folder_path}")
            makedirs(folder_path)
    
    paths_dict['trained_ckpt_path']   = os.path.join(paths_dict['open_active_save_dir'],'ckpt.pt')
    paths_dict['query_result_path']   = os.path.join(paths_dict['finetuned_dir']   ,'query_result.pt')
    paths_dict['finetuned_ckpt_path'] = os.path.join(paths_dict['finetuned_dir']   ,'ckpt.pt')
    paths_dict['test_result_path']    = os.path.join(paths_dict['finetuned_dir']   ,'test_result.pt')
    
    paths_dict['open_result_paths'] = {}
    paths_dict['open_result_roc_paths'] = {}
    paths_dict['open_result_goscr_paths'] = {}
    for key in paths_dict['test_dirs']:
        paths_dict['open_result_paths'][key]       = os.path.join(paths_dict['test_dirs'][key],'open_result.pt')
        paths_dict['open_result_roc_paths'][key]   = os.path.join(paths_dict['test_dirs'][key],"roc.png")
        paths_dict['open_result_goscr_paths'][key] = os.path.join(paths_dict['test_dirs'][key],'goscr.png')
    return paths_dict

def get_budget_list_from_config(config, makedir=True):
    return get_budget_list(config.data)

def get_budget_list(data):
    """Return the list of budget candidates for active learning experiments.
        Args:
            data (str) - dataset name
        Returns:
            budget_list (list) : List of int
    """
    if data in ['CIFAR10']:
        return [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    elif data in ['CIFAR100']:
        return [0, 3000, 6000, 9000, 12000, 15000]
    else:
        raise NotImplementedError()
                           
def prepare_active_learning_dir_from_config(config, budget_list, makedir=True):
    return prepare_active_learning_dir(budget_list,
                                       config.active_save_path,
                                       config.data_download_path,
                                       config.active_save_dir,
                                       config.data,
                                       config.data_config,
                                       config.data_rand_seed,
                                       config.training_method,
                                       config.train_mode,
                                       config.query_method,
                                       config.active_query_scheme,
                                       makedir=makedir)

def prepare_active_learning_dir(budget_list,
                                active_save_path,
                                download_path,
                                active_save_dir,
                                data,
                                data_config,
                                data_rand_seed,
                                training_method,
                                train_mode,
                                query_method,
                                active_query_scheme,
                                # active_val_mode,
                                makedir=True):
    """Return a dictionary of save_paths for active learning
    """
    paths_dict = {}
    paths_dict['data_download_path'] = download_path
    paths_dict['data_save_path'] = get_dataset_info_path(active_save_path,
                                                            data,
                                                            data_config,
                                                            data_rand_seed)
    paths_dict['trainset_info_path'] = get_trainset_info_path(active_save_path, data)
    
    # Where the training/testing results will be saved
    paths_dict['active_save_dir'] = get_active_save_dir(active_save_dir,
                                                        data,
                                                        data_config,
                                                        data_rand_seed,
                                                        training_method,
                                                        train_mode,
                                                        active_query_scheme,
                                                        # active_val_mode,
                                                        makedir=makedir)

    folder_path = paths_dict["active_save_dir"]
    if not os.path.exists(folder_path) and makedir:
        print(f"Make a new folder at: {folder_path}")
        makedirs(folder_path)

    paths_dict["active_query_results"] = {}
    paths_dict["active_ckpt_results"] = {}
    paths_dict["active_test_results"] = {}
    for b in budget_list:
        b_dir = os.path.join(folder_path, str(b), "active_"+query_method)
        if not os.path.exists(b_dir) and makedir:
            print(f"Make a new folder at: {b_dir}")
            makedirs(b_dir)
        paths_dict["active_query_results"][b] = os.path.join(b_dir, "query_result.pt")
        paths_dict["active_ckpt_results"][b]  = os.path.join(b_dir, "ckpt.pt")
        paths_dict["active_test_results"][b]  = os.path.join(b_dir, "test_result.pt")
        
    return paths_dict


def prepare_open_set_learning_dir_from_config(config, makedir=True):
    open_set_methods = global_setting.OPEN_SET_METHOD_DICT[config.training_method]
    return prepare_open_set_learning_dir(config.open_set_save_path,
                                         config.data_download_path,
                                         config.open_save_dir,
                                         config.data,
                                         config.data_config,
                                         config.data_rand_seed,
                                         config.training_method,
                                         config.train_mode,
                                         open_set_methods,
                                         makedir=makedir)

def prepare_open_set_learning_dir(open_set_save_path,
                                  download_path,
                                  open_save_dir,
                                  data,
                                  data_config,
                                  data_rand_seed,
                                  training_method,
                                  train_mode,
                                  open_set_methods,
                                  makedir=True):
    """Return a dictionary of save_paths for open set learning
    """
    paths_dict = {}
    paths_dict['data_download_path'] = download_path
    paths_dict['data_save_path'] = get_dataset_info_path(open_set_save_path,
                                                            data,
                                                            data_config,
                                                            data_rand_seed)
    paths_dict['trainset_info_path'] = get_trainset_info_path(open_set_save_path, data)
    
    
    # Where the training/testing results will be saved
    paths_dict['open_save_dir'] = get_open_save_dir(open_save_dir,
                                                            data,
                                                            data_config,
                                                            data_rand_seed,
                                                            training_method,
                                                            train_mode,
                                                            makedir=makedir)

    folder_path = paths_dict["open_save_dir"]
    if not os.path.exists(folder_path) and makedir:
        print(f"Make a new folder at: {folder_path}")
        makedirs(folder_path)

    paths_dict['trained_ckpt_path']   = os.path.join(paths_dict['open_save_dir']   ,'ckpt.pt')
    paths_dict['test_result_path']    = os.path.join(paths_dict['open_save_dir']   ,'test_result.pt')
    
    paths_dict['test_dirs'] = {}
    for open_set_method in open_set_methods:
        paths_dict['test_dirs'][open_set_method] = os.path.join(paths_dict['open_save_dir'],
                                                                "_".join(["openset", open_set_method]))

    for key in paths_dict['test_dirs']:
        folder_path = paths_dict['test_dirs'][key]
        if not os.path.exists(folder_path) and makedir:
            print(f"Make a new folder at: {folder_path}")
            makedirs(folder_path)
    
    paths_dict['open_result_paths'] = {}
    paths_dict['open_result_roc_paths'] = {}
    paths_dict['open_result_goscr_paths'] = {}
    for key in paths_dict['test_dirs']:
        paths_dict['open_result_paths'][key]       = os.path.join(paths_dict['test_dirs'][key],'open_result.pt')
        paths_dict['open_result_roc_paths'][key]   = os.path.join(paths_dict['test_dirs'][key],"roc.png")
        paths_dict['open_result_goscr_paths'][key] = os.path.join(paths_dict['test_dirs'][key],'goscr.png')
    return paths_dict

