import os, sys
import torch
from torch.utils.data import DataLoader

class SetPrintMode:
    def __init__(self, hidden=False):
        self.hidden = hidden

    def __enter__(self):
        if self.hidden:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        if self.hidden:
            sys.stdout = self._original_stdout

def makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        print(f"{dir_name} already exists.")

def get_checkpoint(round, s_train, seen_classes, trainer, logger):
    return {'round' : round,
            'trainer_checkpoint' : trainer.get_checkpoint(),
            's_train' : s_train,
            'seen_classes' : seen_classes,
            'logger_checkpoint' : logger.get_checkpoint()}

def save_checkpoint(ckpt_dir, checkpoint, epoch=0):
    torch.save(checkpoint,
               os.path.join(ckpt_dir, 
                            'checkpoint_{}_{}.pth.tar'.format(checkpoint['round'],
                                                              epoch)
               ))

def get_subset_dataloaders(dataset, samples, target_transform, batch_size, workers=0, shuffle=True):
    """ Return a dict of dataloaders. Key 'train' refers
    to the train set loader
    """
    dataloaders = {}
    train_subset = torch.utils.data.Subset(dataset, samples)
    train_subset.dataset.target_transform = target_transform
    train_loader = DataLoader(
                       train_subset,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       num_workers=workers,
                       pin_memory=True,
                   )
    dataloaders['train'] = train_loader
    return dataloaders

def get_test_loader(dataset, target_transform, batch_size, workers=0):
    """ Return a dict of dataloaders. Key 'train' refers
    to the train set loader
    """
    dataset.target_transform = target_transform
    test_loader = DataLoader(
                      dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=workers,
                      pin_memory=True,
                  )
    return test_loader

def get_experiment_name(config):
    name = [config.data]
    name += ['rounds', str(config.max_rounds), 'budget', str(config.budget), 'init', config.init_mode]
    name += [config.label_picker]

    if config.label_picker == "uncertainty_measure":
        name += [config.uncertainty_measure]
    else:
        raise NotImplementedError()

    name += ['baseline', config.trainer]
    name += [config.arch, 'pretrained', str(config.pretrained)]
    name += ["lr", str(config.lr), config.optim, "mt", str(config.momentum), "wd", str(config.wd)]
    name += ["epoch", str(config.epochs), "batch", str(config.batch)]
    if config.lr_decay_step != None:
        name += ['lrdecay', str(config.lr_decay_ratio), 'per', str(config.lr_decay_step)]

    if config.trainer == "network":
        name += ['openset_softmax', config.network_eval_mode, str(config.network_eval_threshold)]
    else:
        raise NotImplementedError()
    name = "_".join(name)

    return name
