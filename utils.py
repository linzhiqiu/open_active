import os
import torch
from torch.utils.data import DataLoader

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
    import pdb; pdb.set_trace()  # breakpoint 3217d269 //
    return "test_run"
