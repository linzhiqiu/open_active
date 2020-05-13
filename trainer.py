from trainer_machine import BinarySoftmaxNetwork, Network, OSDNNetwork, OSDNNetworkModified, ClusterNetwork, SigmoidNetwork
from gan import GANFactory
from c2ae import C2AE
from learning_loss import NetworkLearningLoss, get_learning_loss_class
from icalr import ICALR, ICALROSDNNetwork, ICALROSDNNetworkModified, ICALRBinarySoftmaxNetwork
from label_picker import UncertaintyMeasure, CoresetMeasure


class TrainingInstance(object):
    """ A data class holding all resources for training
    """
    def __init__(self, train_dataset, train_samples, open_samples, train_labels, classes, open_classes):
        super(TrainingInstance, self).__init__()
        self.train_dataset = train_dataset # PyTorch train dataset
        self.train_samples = train_samples # List of indices in train set
        self.open_samples = open_samples # List of indices representing training samples belonging to open class
        self.query_samples = train_samples.difference(open_samples) # List of indices representing the unlabeled pool
        self.train_labels = train_labels # List of labels of all train samples
        self.classes = classes # Set of all classes
        self.open_classes = open_classes # Set of all hold out open classes
        self.query_classes = classes.difference(open_classes) # Set of all classes in unlabeled pool


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

        self.trainer_machine = self._init_trainer_machine() # TrainerMachine implements the actual algorithm
        self.label_picker = self._init_label_picker() # LabelPicker implements the active learning query algorithm
    
    def train_then_eval(self, discovered_samples, discovered_classes, test_dataset, eval_verbose=False):
        # Start a new round. Each round has 2 steps:
        # (1) Train the model using discovered_samples from discovered_classes.
        # (2) Then eval on test_dataset.
        return self.trainer_machine.train_then_eval(discovered_samples,
                                                    discovered_classes,
                                                    test_dataset,
                                                    eval_verbose=eval_verbose)

    def get_thresholds_checkpoints(self):
        return self.trainer_machine.get_thresholds_checkpoints()

    def get_exemplar_set(self):
        return self.trainer_machine.exemplar_set
        
    # def train_new_round(self, discovered_samples, discovered_classes):
    #     return self.trainer_machine.train_new_round(discovered_samples, discovered_classes)

    def select_new_data(self, discovered_samples, discovered_classes):
        return self.label_picker.select_new_data(discovered_samples, discovered_classes)

    # def eval(self, test_dataset, verbose=False):
    #     return self.trainer_machine.eval(test_dataset, verbose=verbose)

    def _init_trainer_machine(self):
        """ Initialize all necessary models/optimizer/learning rate scheduler 
        """
        if self.config.trainer == 'network':
            trainer_machine_class = Network
        elif self.config.trainer in ['osdn']:
            trainer_machine_class = OSDNNetwork
        elif self.config.trainer in ['osdn_modified']:
            trainer_machine_class = OSDNNetworkModified
        elif self.config.trainer in ['icalr_osdn', 'icalr_osdn_neg']:
            trainer_machine_class = ICALROSDNNetwork
        elif self.config.trainer in ['icalr_osdn_modified', 'icalr_osdn_modified_neg']:
            trainer_machine_class = ICALROSDNNetworkModified
        elif self.config.trainer == 'c2ae':
            trainer_machine_class = C2AE
        elif self.config.trainer == 'cluster':
            trainer_machine_class = ClusterNetwork
        elif self.config.trainer == 'sigmoid':
            trainer_machine_class = SigmoidNetwork
        elif self.config.trainer == 'binary_softmax':
            trainer_machine_class = BinarySoftmaxNetwork
        elif self.config.trainer == 'icalr_binary_softmax':
            trainer_machine_class = ICALRBinarySoftmaxNetwork
        elif self.config.trainer == 'gan':
            gan_factory = GANFactory(self.config)
            trainer_machine_class = gan_factory.gan_class()
        elif self.config.trainer == 'network_learning_loss':
            trainer_machine_class = NetworkLearningLoss
        elif self.config.trainer == 'icalr':
            trainer_machine_class = ICALR
        elif self.config.trainer == 'icalr_learning_loss':
            trainer_machine_class = get_learning_loss_class(ICALR)
        else:
            raise NotImplementedError()
        return trainer_machine_class(self.config,
                                     self.train_instance)

    def _init_label_picker(self):
        if self.config.trainer in ['network', 'osdn', 'osdn_modified', 'cluster', 'gan', 'sigmoid', 'binary_softmax', 'c2ae', 'network_learning_loss', 'icalr_osdn_neg', 'icalr_osdn_modified_neg',
                                   'icalr', 'icalr_learning_loss', 'icalr_osdn', 'icalr_osdn_modified', 'icalr_binary_softmax']:
            if self.config.label_picker == 'uncertainty_measure':
                label_picker_class = UncertaintyMeasure
            elif self.config.label_picker == 'coreset_measure':
                label_picker_class = CoresetMeasure
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return label_picker_class(self.config,
                                  self.train_instance,
                                  self.trainer_machine)
