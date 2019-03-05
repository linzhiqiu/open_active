from trainer_machine import Network
from label_picker import UncertaintyMeasure

def get_trainer(config, train_dataset, train_samples, train_labels, classes):
    return Trainer(config, train_dataset, train_samples, train_labels, classes)


class TrainingInstance(object):
    """ A data class holding all resources for training
    """
    def __init__(self, train_dataset, train_samples, train_labels, classes):
        super(TrainingInstance, self).__init__()
        self.train_dataset = train_dataset
        self.train_samples = train_samples
        self.train_labels = train_labels
        self.classes = classes


class Trainer(object):
    def __init__(self, config, train_dataset, train_samples, train_labels, classes):
        super(Trainer, self).__init__()
        self.config = config
        self.train_instance = TrainingInstance(train_dataset, 
                                               train_samples,
                                               train_labels,
                                               classes)

        self.trainer_machine = self._init_trainer_machine()
        self.label_picker = self._init_label_picker()
        
    def train_new_round(self, s_train, seen_classes):
        self.trainer_machine.train_new_round(s_train, seen_classes)

    def select_new_data(self, s_train, seen_classes):
        return self.label_picker.select_new_data(s_train, seen_classes)

    def eval(self, test_dataset, seen_classes):
        return self.trainer_machine.eval(test_dataset, seen_classes)

    def get_checkpoint(self):
        """ Return a dictionary of all necessary gadgets in order to resume training
        """
        return {
            'trainer_machine' : self.trainer_machine.get_checkpoint(),
            'label_picker' : self.label_picker.get_checkpoint(),
        }

    def load_checkpoint(self, trainer_checkpoint):
        """ Load from a dictionary
        """
        self.trainer_machine.load_checkpoint(trainer_checkpoint['trainer_machine'])
        self.label_picker.load_checkpoint(trainer_checkpoint['label_picker'])

    def _init_trainer_machine(self):
        """ Initialize all necessary models/optimizer/learning rate scheduler 
        """
        if self.config.trainer == 'network':
            trainer_machine_class = Network
        else:
            raise NotImplementedError()
        return trainer_machine_class(self.config,
                                     self.train_instance)

    def _init_label_picker(self):
        if self.config.label_picker == 'uncertainty_measure':
            label_picker_class = UncertaintyMeasure
        else:
            raise NotImplementedError()
        return label_picker_class(self.config,
                                  self.train_instance,
                                  self.trainer_machine)
