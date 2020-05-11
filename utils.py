import os, sys
import torch
from torch.utils.data import DataLoader

import global_setting # include some constant value

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

class FixedRepresentationDataset(torch.utils.data.TensorDataset):
    def __init__(self, data_tensor, target_tensor):
        assert (data_tensor.size(0) == target_tensor.size(0))
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, idx):
        return self.data_tensor[idx, ...], self.target_tensor[idx, ...] 

    def __len__(self):
        return self.data_tensor.size(0)

def makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        print(f"{dir_name} already exists.")

def get_checkpoint(round, discovered_samples, open_examples, discovered_classes, open_classes, trainer, logger):
    return {'round' : round,
            'trainer_checkpoint' : trainer.get_checkpoint(),
            'discovered_samples' : discovered_samples,
            'open_examples' : open_examples,
            'discovered_classes' : discovered_classes,
            'open_classes' : open_classes,
            'logger_checkpoint' : logger.get_checkpoint()}

def save_checkpoint(ckpt_dir, checkpoint, epoch=0):
    torch.save(checkpoint,
               os.path.join(ckpt_dir, 
                            'checkpoint_{}_{}.pth.tar'.format(checkpoint['round'],
                                                              epoch)
               ))

def get_loader(dataset, target_transform, batch_size=1, shuffle=False, workers=0):
    # TODO: Affirm that changing dataset.target_transform is okay.
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
    """ Return a dict of train dataloaders.
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

def get_target_mapping_func(classes, discovered_classes, open_classes):
    """ Return a function that map discovered_classes indices to 0-len(discovered_classes). 
        If not in hold-out open classes, undiscovered classes
        are mapped to -1. Hold-out open classes are mapped to -2.
        Always return the same indices as long as discovered classes is the same.
        Args:
            classes: The list of all classes
            discovered_classes: The set of all discovered classes
            open_classes: The set of all hold-out open classes
    """

    discovered_classes = sorted(list(discovered_classes))
    open_classes = sorted(list(open_classes))
    mapping = {idx : global_setting.OPEN_CLASS_INDEX if idx in open_classes else 
                     global_setting.UNDISCOVERED_CLASS_INDEX if idx not in discovered_classes else 
                     discovered_classes.index(idx)
               for idx in classes}
    return lambda idx : mapping[idx]

def get_target_mapping_func_for_tensor(classes, discovered_classes, open_classes, device='cuda'):
    """ Exactly the same as get_target_mapping_func but operate on tensor level
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
    """ Return a dictionary that map 0-len(discovered_classes) to true discovered_classes indices.
        Always return the same indices as long as discovered classes (which is a set) is the same.
        Args:
            classes: The list of all classes
            discovered_classes: The set of all discovered classes
    """
    discovered_classes = sorted(list(discovered_classes))
    mapping = {idx : global_setting.UNDISCOVERED_CLASS_INDEX if idx not in discovered_classes else discovered_classes.index(idx)
               for idx in classes}
    unmapping = {mapping[true_index] : true_index for true_index in mapping.keys()}
    if -1 in unmapping.keys():
        del unmapping[-1]
    return unmapping

def get_target_unmapping_func_for_list(classes, discovered_classes):
    discovered_classes = sorted(list(discovered_classes))
    mapping = {idx : global_setting.UNDISCOVERED_CLASS_INDEX if idx not in discovered_classes else discovered_classes.index(idx)
               for idx in classes}
    unmapping_dict = {mapping[true_index] : true_index for true_index in mapping.keys()}
    unmapping_dict[global_setting.UNDISCOVERED_CLASS_INDEX] = global_setting.UNDISCOVERED_CLASS_INDEX
    def unmapp_func(lst):
        return list(map(lambda x: unmapping_dict[x], lst))
    return unmapp_func

def enable_graphite(config):
    import os
    config.data_path = os.path.join("/scratch/datasets", config.data_path)

def get_data_param(config):
    # For first round thresholds values logging
    return "_".join([config.data, config.init_mode])

def get_data_active_param(config):
    # For learning loss active learning logging
    name = [config.data, config.init_mode, "r", str(config.max_rounds), "b", str(config.budget)]
    name += ['retrain', str(config.icalr_mode), str(config.icalr_retrain_threshold), str(config.icalr_retrain_criterion)]
    name += ['exemplar', str(config.icalr_exemplar_size)]

    name += ['icalr', str(config.icalr_strategy)]
    if config.icalr_strategy == 'naive':
        name += ['mode', str(config.icalr_naive_strategy)]
    elif config.icalr_strategy == 'smooth':
        name += ['smooth_eps', str(config.smooth_epochs)]
    elif config.icalr_strategy == 'proto':
        name += ['mode', str(config.icalr_proto_strategy)]
    
    return "_".join(name)

def get_method_param(config):
    # For first round thresholds values logging
    if config.trainer in ['network','icalr','binary_softmax']:
        setting_str = config.threshold_metric
    elif config.trainer == 'sigmoid':
        setting_str = config.sigmoid_train_mode
    elif config.trainer == "icalr_binary_softmax":
        setting_str = config.icalr_binary_softmax_train_mode
    elif config.trainer == 'c2ae':
        setting_str = config.c2ae_train_mode
    elif config.trainer in ['osdn_modified', 'osdn', 'icalr_osdn_modified', 'icalr_osdn', 'icalr_osdn_modified_neg', 'icalr_osdn_neg']:
        setting_str = config.distance_metric
    elif config.trainer in ['cluster']:
        setting_str = "_".join([config.clustering, "dist", config.distance_metric, "metric", config.threshold_metric])
    elif config.trainer in ['network_learning_loss', 'icalr_learning_loss']:
        setting_str = "_".join([config.threshold_metric, 'mode', config.learning_loss_train_mode, 'lmb', str(config.learning_loss_lambda),
                                'margin', str(config.learning_loss_margin),
                                'start_ep', str(config.learning_loss_start_epoch),
                                'stop_ep', str(config.learning_loss_stop_epoch)])
    else:
        raise NotImplementedError()
    return "_".join([config.trainer, setting_str])

def get_active_param(config):
    # For active learning acc logging    
    name = [config.label_picker]
    if config.label_picker == "uncertainty_measure":
        name += [config.uncertainty_measure, "s", config.active_random_sampling]
    elif config.label_picker == "coreset_measure":
        name += [config.coreset_measure, "s", config.active_random_sampling, config.coreset_feature]
    else:
        raise NotImplementedError()
    name += ['oa', config.open_active_setup]
    return "_".join(name)

def get_experiment_name(config):
    name_str = ''

    name = []
    name += [config.data]
    name += ['rounds', str(config.max_rounds), 'budget', str(config.budget), 'init', config.init_mode]
    name += ['retrain', str(config.icalr_mode), str(config.icalr_retrain_threshold), str(config.icalr_retrain_criterion)]
    name += ['exemplar', str(config.icalr_exemplar_size)]
    name_str += "_".join(name) + os.sep

    name = []
    name += ['icalr', str(config.icalr_strategy)]
    if config.icalr_strategy == 'naive':
        name += ['mode', str(config.icalr_naive_strategy)]
    elif config.icalr_strategy == 'proto':
        name += ['mode', str(config.icalr_proto_strategy)]
    elif config.icalr_strategy == 'smooth':
        name += ['smooth_eps', str(config.smooth_epochs)]
    name_str += "_".join(name) + os.sep
    
    name = []

    if config.label_picker == "uncertainty_measure":
        name += ["uncertain"]
        name += [config.uncertainty_measure, config.active_random_sampling]
    elif config.label_picker == "coreset_measure":
        name += ["coreset"]
        name += [config.coreset_measure, config.active_random_sampling, config.coreset_feature]
    else:
        raise NotImplementedError()
    name += ['oa', config.open_active_setup]
    name_str += "_".join(name) + os.sep

    name = []
    if config.trainer == 'gan':
        name += ['gan', config.gan_player, 'mode', config.gan_mode, 'setup', config.gan_setup]
        if config.gan_player == 'multiple':
            name += ['multi', config.gan_multi]
    elif config.trainer in ["network", 'icalr']:
        name += ['openset', config.threshold_metric, config.network_eval_mode, str(config.network_eval_threshold)]
    elif config.trainer in ["network_learning_loss", 'icalr_learning_loss']:
        name += ['learning_loss', config.threshold_metric, config.network_eval_mode, str(config.network_eval_threshold), 
                 # 'mode', config.learning_loss_train_mode, 
                 'lmb', str(config.learning_loss_lambda),
                 'margin', str(config.learning_loss_margin),
                 'start_ep', str(config.learning_loss_start_epoch),
                 'stop_ep', str(config.learning_loss_stop_epoch)]
    elif config.trainer == "sigmoid":
        name += ["sigmoid", config.sigmoid_train_mode, config.network_eval_mode, str(config.network_eval_threshold)]
    elif config.trainer == "binary_softmax":
        name += [config.network_eval_mode, str(config.network_eval_threshold)]
    elif config.trainer == "icalr_binary_softmax":
        name += [config.icalr_binary_softmax_train_mode, config.network_eval_mode, str(config.network_eval_threshold)]
    elif config.trainer == "c2ae":
        name += ["c2ae", config.c2ae_train_mode, "alpha", str(config.c2ae_alpha), ]
    elif config.trainer in ['osdn','osdn_modified']:
        name += ['osdn' if not config.trainer == 'osdn_modified' else 'osdnmod', 
                 "dist", config.distance_metric]
        if 'eu' in config.distance_metric:
            name += ['diveu', str(config.div_eu)]
        if config.pseudo_open_set == None:
            # Using fixed hyper
            name += ["thre", str(config.osdn_eval_threshold),
                     "alpha", config.alpha_rank,
                     "tail", config.weibull_tail_size]
        else:
            # Using cross validation/meta learning to decide hyper
            assert config.openmax_meta_learn != None
            name += ["meta", str(config.openmax_meta_learn)]
        # name += ['mav', config.mav_features_selection]
    elif config.trainer in ['icalr_osdn','icalr_osdn_modified', 'icalr_osdn_neg', 'icalr_osdn_modified_neg']:
        name += ['icalr_openmax' if not config.trainer == 'icalr_osdn_modified' else 'icalr_osdnmod', 
                 "dist", config.distance_metric]
        if 'eu' in config.distance_metric:
            name += ['diveu', str(config.div_eu)]
        if config.pseudo_open_set == None:
            # Using fixed hyper
            name += ["thre", str(config.osdn_eval_threshold),
                     "alpha", config.alpha_rank,
                     "tail", config.weibull_tail_size]
        else:
            # Using cross validation/meta learning to decide hyper
            assert config.openmax_meta_learn != None
            name += ["meta", str(config.openmax_meta_learn)]
        # name += ['mav', config.mav_features_selection]
    elif config.trainer == 'cluster':
        name += ['cluster', config.clustering, 'distance', config.distance_metric]
        if 'eu' in config.distance_metric:
            name += ['div_eu', str(config.div_eu)]
        if config.clustering == 'rbf_train':
            name += ['gamma', str(config.rbf_gamma)]
        if config.pseudo_open_set == None:
            name += ['threshold', str(config.cluster_eval_threshold)]
        else:
            name += ['threshold', 'metalearn']
        name += ['metric', config.threshold_metric]
        name += ['level', config.cluster_level]
    else:
        raise NotImplementedError()

    # if config.trainer in ["network", "osdn", "osdn_modified"]:
    name += ['cw', config.class_weight]
    if config.pseudo_open_set != None:
        name += ['p', str(config.pseudo_open_set), 'r', str(config.pseudo_open_set_rounds), 'm', config.pseudo_open_set_metric]
        if config.pseudo_same_network:
            name += ['same_network']
    name_str += "_".join(name) + os.sep

    name = []
    name += ['baseline', config.trainer]
    name += [config.arch, 'pretrained', str(config.pretrained)]
    name += ["lr", str(config.lr), config.optim, "mt", str(config.momentum), "wd", str(config.wd)]
    name += ["epoch", str(config.epochs), "batch", str(config.batch)]
    if config.lr_decay_step != None:
        name += ['lrdecay', str(config.lr_decay_ratio), 'per', str(config.lr_decay_step)]
    name_str += "_".join(name)

    return name_str
