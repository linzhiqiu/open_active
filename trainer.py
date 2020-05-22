import trainer_machine, query_machine
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
    def __init__(self, training_method, train_mode, trainer_config, trainset_info, query_method, budget, open_set_method, save_dir=None):
        """The main class for training/querying/finetuning
            Args:
                training_method (str) : The method for training the network
                train_mode (str) : Specify the training details, such as lr, batchsize...
                trainer_config (dict) : Dictionary that includes all training hyperparameters
                trainset_info (TrainsetInfo) : The details about the training set
                query_method (str) : The method for querying from the unlabeled pool
                budget (int/float) : The querying budget
                open_set_method (str) : The method for open_set recognition
                save_dir (str) : The directory to save/load the checkpoints.
        """
        super(Trainer, self).__init__()
        self.training_method = training_method
        self.train_mode = train_mode
        self.trainset_info = trainset_info
        self.trainer_config = trainer_config
        self.query_method = query_method
        self.budget = budget
        self.open_set_method = open_set_method

        trainer_save_dir = os.path.join(save_dir,
                                        "_".join([training_method,train_mode]))
        
        query_dir     = os.path.join(trainer_save_dir, "active_"+self.query_method)
        self.finetuned_dir = os.path.join(query_dir, "budget_"+str(budget))
        self.test_dir      = os.path.join(self.finetuned_dir, "openset_"+self.open_set_method)

        for folder in [trainer_save_dir, self.finetuned_dir, self.test_dir]:
            if not os.path.exists(folder):
                print(f"Make a new folder at: {folder}")
                os.makedirs(folder)
       
        self.trained_ckpt_path   = os.path.join(trainer_save_dir,'ckpt.pt')
        self.query_result_path   = os.path.join(self.finetuned_dir,'query_result.pt')
        self.finetuned_ckpt_path = os.path.join(self.finetuned_dir,'ckpt.pt')
        self.test_result_path    = os.path.join(self.test_dir,'test_result.pt')

        self.trainer_machine = trainer_machine.get_trainer_machine(training_method,
                                                                   trainset_info,
                                                                   trainer_config)
        self.query_machine = query_machine.get_query_machine(query_method,
                                                             trainset_info,
                                                             trainer_config)

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

    def eval(self, discovered_classes, test_dataset):
        assert len(discovered_classes) == len(self.trainset_info.query_classes)
        raise NotImplementedError()