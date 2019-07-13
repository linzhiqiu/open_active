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

def makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        print(f"{dir_name} already exists.")

def get_checkpoint(round, s_train, open_examples, seen_classes, open_classes, trainer, logger):
    return {'round' : round,
            'trainer_checkpoint' : trainer.get_checkpoint(),
            's_train' : s_train,
            'open_examples' : open_examples,
            'seen_classes' : seen_classes,
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

def get_target_mapping_func(classes, seen_classes, open_classes):
    """ Return a function that map seen_classes indices to 0-len(seen_classes). 
        If not in hold-out open classes, unseen classes
        are mapped to -1. Hold-out open classes are mapped to -2.
        Always return the same indices as long as seen classes is the same.
        Args:
            classes: The list of all classes
            seen_classes: The set of all seen classes
            open_classes: The set of all hold-out open classes
    """
    seen_classes = sorted(list(seen_classes))
    open_classes = sorted(list(open_classes))
    mapping = {idx : global_setting.OPEN_CLASS_INDEX if idx in open_classes else 
                     global_setting.UNSEEN_CLASS_INDEX if idx not in seen_classes else 
                     seen_classes.index(idx)
               for idx in classes}
    return lambda idx : mapping[idx]

def get_target_unmapping_dict(classes, seen_classes):
    """ Return a dictionary that map 0-len(seen_classes) to true seen_classes indices.
        Always return the same indices as long as seen classes (which is a set) is the same.
        Args:
            classes: The list of all classes
            seen_classes: The set of all seen classes
    """
    seen_classes = sorted(list(seen_classes))
    mapping = {idx : -1 if idx not in seen_classes else seen_classes.index(idx)
               for idx in classes}
    unmapping = {mapping[true_index] : true_index for true_index in mapping.keys()}
    if -1 in unmapping.keys():
        del unmapping[-1]
    return unmapping

def get_experiment_name(config):
    name_str = ''

    name = []
    name += [config.data]
    name += ['rounds', str(config.max_rounds), 'budget', str(config.budget), 'init', config.init_mode]
    name_str += "_".join(name) + os.sep
    
    name = []
    name += [config.label_picker]

    if config.label_picker == "uncertainty_measure":
        name += [config.uncertainty_measure]
    else:
        raise NotImplementedError()
    name_str += "_".join(name) + os.sep

    name = []
    if config.trainer == "network":
        name += ['openset', config.threshold_metric, config.network_eval_mode, str(config.network_eval_threshold)]
    elif config.trainer in ['osdn','osdn_modified']:
        name += ['osdn_openmax' if not config.trainer == 'osdn_modified' else 'osdn_modified', 
                 "distance", config.distance_metric,]
        if config.pseudo_open_set == None:
            # Using fixed hyper
            name += ["threshold", str(config.osdn_eval_threshold),
                     "alpha", config.alpha_rank,
                     "tailsize", config.weibull_tail_size]
        else:
            # Using cross validation/meta learning to decide hyper
            assert config.openmax_meta_learn != None
            name += ["metalearn", str(config.openmax_meta_learn)]
        name += ['mav', config.mav_features_selection]
    elif config.trainer == 'cluster_network':
        name += ['cluster_network', config.clustering, config.network_eval_mode, str(config.network_eval_threshold)]
    else:
        raise NotImplementedError()
    name += ['classweight', config.class_weight]
    if config.pseudo_open_set != None:
        name += ['newpseopen', str(config.pseudo_open_set), 'round', str(config.pseudo_open_set_rounds)]
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
