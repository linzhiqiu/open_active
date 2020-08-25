import trainer_machine, query_machine, eval_machine
import os


class Trainer(object):
    def __init__(self, training_method, trainer_config, dataset_info):
        """The highest-level class performing training/querying/finetuning
        It will instantitate TrainerMachine/QueryMachine/EvalMachine objects based on the arguments,
        and use them for training/querying/testing.

        Args:
            training_method (str) : The network training method
            trainer_config (train_config.TrainerConfig) : Including all training hyperparameters
            dataset_info (dataset_factory.DatasetInfo) : The details about the dataset set
        """
        super(Trainer, self).__init__()
        self.training_method = training_method
        self.dataset_info = dataset_info
        self.trainer_config = trainer_config

        self.trainer_machine = trainer_machine.get_trainer_machine(
                                   training_method,
                                   dataset_info,
                                   trainer_config
                               )
        
    def train(self, discovered_samples, discovered_classes, ckpt_path, verbose=False):
        """Performs training using [discovered_samples] and saved the results to [ckpt_path]

        Args:
            discovered_samples (list[int]): All discovered (labeled) training samples
            discovered_classes (list[int]): All classes with discovered samples
            ckpt_path ([type]): Where the training results will be saved
            verbose (bool, optional): Whether to print more information. Defaults to False
        """        
        self.trainer_machine.train(
            discovered_samples,
            discovered_classes,
            ckpt_path=ckpt_path,
            verbose=verbose
        )
    
    def query(self,
              discovered_samples,
              discovered_classes,
              budget,
              query_method,
              query_result_path,
              verbose=False):
        """Performs querying from unlabeled pool.

        Args:
            discovered_samples (list[int]): All discovered (labeled) training samples
            discovered_classes (list[int]): All classes with discovered samples
            budget (int): Number of new samples to label
            query_method (str) : The method for querying from the unlabeled pool
            query_result_path (str): Where the result will be saved
            verbose (bool, optional): Whether to print more information. Defaults to False

        Returns:
            list[int]: Discovered (labeled) training samples after querying
            list[int]: Classes with discovered samples after querying
        """
        query_machine_instance = query_machine.get_query_machine(
            query_method,
            self.dataset_info,
            self.trainer_config
        )
        return query_machine_instance.query(
            self.trainer_machine,
            budget,
            discovered_samples,
            discovered_classes,
            query_result_path=query_result_path,
            verbose=verbose
        )
    
    def eval_closed_set(self, discovered_classes, result_path, verbose=False):
        """Evaluating on test set (assuming all classes are discovered, so no open set method is used).
        For test samples not in discovered classes, their accuracy is zero.

        Args:
            discovered_classes (list[int]): All classes with discovered samples
            result_path ([type]): Where the result will be saved
            verbose (bool, optional): Whether to print more information. Defaults to False

        Returns:
            float: Test Accuracy on all test samples. For samples not in discovered set of classes,
                   they are always marked as incorrect prediction.
        """        
        return self.trainer_machine.eval_closed_set(
            discovered_classes,
            result_path=result_path,
            verbose=verbose
        )

    def eval_open_set(self,
                      discovered_samples,
                      discovered_classes,
                      open_set_methods,
                      result_paths,
                      roc_paths,
                      goscr_paths,
                      verbose=False):
        """Evaluating on test set using all open set set methods in 'open_set_methods'(list).

        Args:
            discovered_samples (list[int]): All discovered (labeled) training samples
            discovered_classes (list[int]): All classes with discovered samples
            open_set_methods (list[str]): All open set methods to evaluate
            result_paths (list[str]): Where the evaluation result will be saved
            roc_paths (list[str]): Where the ROC result will be saved
            goscr_paths (list[str]): Where the GOSCR result will be saved
            verbose (bool, optional): Whether to print more information. Defaults to False
        """        
        for open_set_method in open_set_methods:
            eval_machine_instance = eval_machine.get_eval_machine(
                open_set_method,
                self.trainer_machine,
                self.dataset_info,
                self.trainer_config,
            )
            eval_machine_instance.eval_open_set(
                discovered_samples,
                discovered_classes,
                result_path=result_paths[open_set_method],
                roc_path=roc_paths[open_set_method],
                goscr_path=goscr_paths[open_set_method],
                verbose=verbose
            )
        

class ActiveTrainer(object):
    def __init__(self, training_method, active_config, dataset_info, query_method, test_dataset, val_samples=None):
        """The main class for training/querying/finetuning
            Args:
                training_method (str) : The method for training the network
                active_config (dict) : Dictionary that includes all training hyperparameters
                dataset_info (DatasetInfo) : The details about the dataset set
                query_method (str) : The method for querying from the unlabeled pool
                test_dataset (torch.nn.Dataset) : The test dataset
        """
        super(ActiveTrainer, self).__init__()
        self.training_method = training_method
        self.dataset_info = dataset_info
        self.active_config = active_config
        self.query_method = query_method

        self.trainer_machine = trainer_machine.get_trainer_machine(training_method,
                                                                   dataset_info,
                                                                   active_config)
        self.query_machine = query_machine.get_query_machine(query_method,
                                                             dataset_info,
                                                             active_config)

    def train(self, discovered_samples, discovered_classes, trained_ckpt_path, verbose=False):
        """Performs training using discovered_samples
        """
        self.trainer_machine.train(
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
    
    # def eval_closed_set(self, discovered_classes, test_dataset, test_result_path, verbose=False):
    #     return self.trainer_machine.eval_closed_set(discovered_classes,
    #                                                 test_dataset,
    #                                                 result_path=test_result_path,
    #                                                 verbose=verbose)
    def eval_closed_set(self, discovered_classes, test_result_path, verbose=False):
        return self.trainer_machine.eval_closed_set(discovered_classes,
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


class OpenTrainer(object):
    def __init__(self,
                 training_method,
                 open_set_config,
                 dataset_info,
                 open_set_methods,
                 test_dataset,
                 val_samples,
                 paths_dict):
        """The main class for training/querying/finetuning
            Args:
                training_method (str) : The method for training the network
                open_set_config (dict) : Dictionary that includes all training hyperparameters
                dataset_info (DatasetInfo) : The details about the dataset set
                open_set_methods (list) : The list of methods for open_set recognition
                paths_dict (dict) : Dictionary of output paths
        """
        super(OpenTrainer, self).__init__()
        self.training_method = training_method
        self.open_set_config = open_set_config
        self.open_set_methods = open_set_methods
        self.dataset_info = dataset_info
        self.paths_dict = paths_dict

        self.trained_ckpt_path    = paths_dict['trained_ckpt_path']
        self.test_result_path     = paths_dict['test_result_path']
        self.open_result_paths    = paths_dict['open_result_paths']
        self.roc_result_paths     = paths_dict['open_result_roc_paths']
        self.goscr_result_paths   = paths_dict['open_result_goscr_paths']

        self.trainer_machine = trainer_machine.get_trainer_machine(training_method,
                                                                   dataset_info,
                                                                   open_set_config)


        self.eval_machines = {}
        for open_set_method in open_set_methods:
            self.eval_machines[open_set_method] = eval_machine.get_eval_machine(
                                                      open_set_method,
                                                      self.trainer_machine,
                                                      dataset_info,
                                                      open_set_config,
                                                      self.roc_result_paths[open_set_method],
                                                      self.goscr_result_paths[open_set_method]
                                                  )

    def train(self, discovered_samples, discovered_classes, verbose=False):
        """Performs training using discovered_samples
        """
        self.trainer_machine.train(
            discovered_samples,
            discovered_classes,
            ckpt_path=self.trained_ckpt_path,
            verbose=verbose
        )
    
    def eval_closed_set(self, discovered_classes, verbose=False):
        return self.trainer_machine.eval_closed_set(discovered_classes,
                                                    result_path=self.test_result_path,
                                                    verbose=verbose)

    def eval_open_set(self, discovered_samples, discovered_classes, verbose=False):
        for open_set_method in self.open_result_paths:
            eval_machine = self.eval_machines[open_set_method]
            eval_machine.eval_open_set(discovered_samples,
                                       discovered_classes,
                                       result_path=self.open_result_paths[open_set_method],
                                       verbose=verbose)
     
