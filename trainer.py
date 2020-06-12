import trainer_machine, query_machine, eval_machine
import os

class TrainsetInfo(object):
    """ A data class holding all resources for training
    """
    def __init__(self, train_dataset, train_samples, open_samples, train_labels, classes, open_classes):
        super(TrainsetInfo, self).__init__()
        self.train_dataset = train_dataset # PyTorch train dataset
        self.train_samples = train_samples # List of indices in train set
        self.open_samples = open_samples # List of indices representing training samples belonging to open class
        self.query_samples = train_samples.difference(open_samples) # List of indices representing the unlabeled pool
        self.train_labels = train_labels # List of labels of all train samples
        self.classes = classes # Set of all classes
        self.open_classes = open_classes # Set of all hold out open classes
        self.query_classes = classes.difference(open_classes) # Set of all classes in unlabeled pool


class Trainer(object):
    def __init__(self, training_method, train_mode, trainer_config, trainset_info, query_method, budget, open_set_methods, paths_dict):
        """The main class for training/querying/finetuning
            Args:
                training_method (str) : The method for training the network
                train_mode (str) : Specify the training details, such as lr, batchsize...
                trainer_config (dict) : Dictionary that includes all training hyperparameters
                trainset_info (TrainsetInfo) : The details about the training set
                query_method (str) : The method for querying from the unlabeled pool
                budget (int/float) : The querying budget
                open_set_methods (list) : The list of methods for open_set recognition
                paths_dict (dict) : The directionary that has directory to save/load the checkpoints.
        """
        super(Trainer, self).__init__()
        self.training_method = training_method
        self.train_mode = train_mode
        self.trainset_info = trainset_info
        self.trainer_config = trainer_config
        self.query_method = query_method
        self.budget = budget
        self.open_set_methods = open_set_methods
       
        self.trained_ckpt_path   = paths_dict['trained_ckpt_path']
        self.query_result_path   = paths_dict['query_result_path']
        self.finetuned_ckpt_path = paths_dict['finetuned_ckpt_path']
        self.test_result_path    = paths_dict['test_result_path']
        self.open_result_paths    = paths_dict['open_result_paths']
        self.roc_result_paths     = paths_dict['open_result_roc_paths']
        self.goscr_result_paths   = paths_dict['open_result_goscr_paths']

        self.trainer_machine = trainer_machine.get_trainer_machine(training_method,
                                                                   train_mode,
                                                                   trainset_info,
                                                                   trainer_config)
        self.query_machine = query_machine.get_query_machine(query_method,
                                                             trainset_info,
                                                             trainer_config)
        self.eval_machines = {}
        for open_set_method in open_set_methods:
            self.eval_machines[open_set_method] = eval_machine.get_eval_machine(
                                                      open_set_method,
                                                      self.trainer_machine,
                                                      trainset_info,
                                                      trainer_config,
                                                      self.roc_result_paths[open_set_method],
                                                      self.goscr_result_paths[open_set_method]
                                                  )

    def train(self, discovered_samples, discovered_classes, verbose=False):
        """Performs training using discovered_samples
        """
        self.trainer_machine.train(discovered_samples,
                                   discovered_classes,
                                   ckpt_path=self.trained_ckpt_path,
                                   verbose=verbose)
    
    def query(self, discovered_samples, discovered_classes, verbose=False):
        """Performs querying from unlabeled pool. discovered_samples is the already labaled samples
        """
        return self.query_machine.query(self.trainer_machine,
                                        self.budget,
                                        discovered_samples,
                                        discovered_classes,
                                        query_result_path=self.query_result_path,
                                        verbose=verbose)
    
    def finetune(self, discovered_samples, discovered_classes, verbose=False):
        self.trainer_machine.finetune(discovered_samples,
                                      discovered_classes,
                                      ckpt_path=self.finetuned_ckpt_path,
                                      verbose=verbose)

    def eval_closed_set(self, discovered_classes, test_dataset, verbose=False):
        return self.trainer_machine.eval_closed_set(discovered_classes,
                                                    test_dataset,
                                                    result_path=self.test_result_path,
                                                    verbose=verbose)

    def eval_open_set(self, discovered_samples, discovered_classes, test_dataset, verbose=False):
        for open_set_method in self.open_result_paths:
            eval_machine = self.eval_machines[open_set_method]
            eval_machine.eval_open_set(discovered_samples,
                                       discovered_classes,
                                       test_dataset,
                                       result_path=self.open_result_paths[open_set_method],
                                       verbose=verbose)
        

class ActiveTrainer(object):
    def __init__(self, training_method, active_train_mode, active_config, trainset_info, query_method, active_val_mode):
        """The main class for training/querying/finetuning
            Args:
                training_method (str) : The method for training the network
                active_train_mode (str) : Specify the training details, such as lr, batchsize...
                active_config (dict) : Dictionary that includes all training hyperparameters
                trainset_info (TrainsetInfo) : The details about the training set
                query_method (str) : The method for querying from the unlabeled pool
                active_val_mode (str or None) : How to select the validation set
        """
        super(ActiveTrainer, self).__init__()
        self.training_method = training_method
        self.active_train_mode = active_train_mode
        self.trainset_info = trainset_info
        self.active_config = active_config
        self.query_method = query_method
        self.active_val_mode = active_val_mode
       
        self.trainer_machine = trainer_machine.get_trainer_machine(training_method,
                                                                   active_train_mode,
                                                                   trainset_info,
                                                                   active_config,
                                                                   val_mode=self.active_val_mode)
        self.query_machine = query_machine.get_query_machine(query_method,
                                                             trainset_info,
                                                             active_config)

    def train(self, discovered_samples, discovered_classes, trained_ckpt_path, verbose=False):
        """Performs training using discovered_samples
        """
        self.trainer_machine.finetune(
            discovered_samples,
            discovered_classes,
            ckpt_path=trained_ckpt_path,
            verbose=verbose
        )
    
    def query(self, b, discovered_samples, discovered_classes, query_result_path, verbose=False):
        """Performs querying from unlabeled pool. discovered_samples is the already labaled samples
        """
        return self.query_machine.query(self.trainer_machine,
                                        b,
                                        discovered_samples,
                                        discovered_classes,
                                        query_result_path=query_result_path,
                                        verbose=verbose)
    
    def eval_closed_set(self, discovered_classes, test_dataset, test_result_path, verbose=False):
        return self.trainer_machine.eval_closed_set(discovered_classes,
                                                    test_dataset,
                                                    result_path=test_result_path,
                                                    verbose=verbose)

    # def eval_open_set(self, discovered_samples, discovered_classes, test_dataset, verbose=False):
    #     for open_set_method in self.open_result_paths:
    #         eval_machine = self.eval_machines[open_set_method]
    #         eval_machine.eval_open_set(discovered_samples,
    #                                    discovered_classes,
    #                                    test_dataset,
    #                                    result_path=self.open_result_paths[open_set_method],
    #                                    verbose=verbose)
     