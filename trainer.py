from trainer_machine import Network, OSDNNetwork, OSDNNetworkModified, ClusterNetwork, SigmoidNetwork
from gan import GANFactory
from c2ae import C2AE
from learning_loss import NetworkLearningLoss
from label_picker import UncertaintyMeasure

def get_trainer(config, train_dataset, train_samples, open_samples, train_labels, classes, open_classes):
    return Trainer(config, train_dataset, train_samples, open_samples, train_labels, classes, open_classes)


class TrainingInstance(object):
    """ A data class holding all resources for training
    """
    def __init__(self, train_dataset, train_samples, open_samples, train_labels, classes, open_classes):
        super(TrainingInstance, self).__init__()
        self.train_dataset = train_dataset
        self.train_samples = train_samples
        self.open_samples = open_samples
        self.query_samples = train_samples.difference(open_samples)
        self.train_labels = train_labels
        self.classes = classes
        self.open_classes = open_classes
        self.query_classes = classes.difference(open_classes)


class Trainer(object):
    def __init__(self, config, train_dataset, train_samples, open_samples, train_labels, classes, open_classes):
        super(Trainer, self).__init__()
        self.config = config
        self.train_instance = TrainingInstance(train_dataset,
                                               train_samples,
                                               open_samples,
                                               train_labels,
                                               classes,
                                               open_classes)

        self.trainer_machine = self._init_trainer_machine()
        self.label_picker = self._init_label_picker()
    
    def train_then_eval(self, s_train, seen_classes, test_dataset, eval_verbose=False):
        # Start a new round. Train the model using s_train from seen_classes. Then eval on test_dataset.
        return self.trainer_machine.train_then_eval(s_train, seen_classes, test_dataset, eval_verbose=eval_verbose)

    def get_thresholds_checkpoints(self):
        return self.trainer_machine.get_thresholds_checkpoints()
        
    # def train_new_round(self, s_train, seen_classes):
    #     return self.trainer_machine.train_new_round(s_train, seen_classes)

    def select_new_data(self, s_train, seen_classes):
        return self.label_picker.select_new_data(s_train, seen_classes)

    # def eval(self, test_dataset, verbose=False):
    #     return self.trainer_machine.eval(test_dataset, verbose=verbose)

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
        elif self.config.trainer == 'osdn':
            trainer_machine_class = OSDNNetwork
        elif self.config.trainer == 'c2ae':
            trainer_machine_class = C2AE
        elif self.config.trainer == 'osdn_modified':
            trainer_machine_class = OSDNNetworkModified
        elif self.config.trainer == 'cluster':
            trainer_machine_class = ClusterNetwork
        elif self.config.trainer == 'sigmoid':
            trainer_machine_class = SigmoidNetwork
        elif self.config.trainer == 'gan':
            gan_factory = GANFactory(self.config)
            trainer_machine_class = gan_factory.gan_class()
        elif self.config.trainer == 'network_learning_loss':
            trainer_machine_class = NetworkLearningLoss
        else:
            raise NotImplementedError()
        return trainer_machine_class(self.config,
                                     self.train_instance)

    def _init_label_picker(self):
        if self.config.trainer in ['network', 'osdn', 'osdn_modified', 'cluster', 'gan', 'sigmoid', 'c2ae', 'network_learning_loss']:
            if self.config.label_picker == 'uncertainty_measure':
                label_picker_class = UncertaintyMeasure
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return label_picker_class(self.config,
                                  self.train_instance,
                                  self.trainer_machine)
