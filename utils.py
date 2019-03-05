import os

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

def get_experiment_name(config):
    raise NotImplementedError()
