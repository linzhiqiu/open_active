

def get_logger(log_name='test', ckpt_dir='./ckpts/test', writer=None):
    return Logger(log_name, ckpt_dir, writer=writer)

class Logger(object):
    """docstring for Logger"""
    def __init__(self, log_name, ckpt_dir, writer=None):
        super(Logger, self).__init__()
        self.log_name = log_name
        self.ckpt_dir = ckpt_dir
        self.writer = writer
        self.round = 0
        self.discovered_samples = None
        self.discovered_classes = None
        self.acc_results_list = []

    def get_checkpoint(self):
        return {'log_name' : self.log_name,
                'ckpt_dir' : self.ckpt_dir,
                'round' : self.round,
                'discovered_samples' : self.discovered_samples,
                'discovered_classes' : self.discovered_classes,
                'open_classes' : self.open_classes,
                'acc_results_list' : self.acc_results_list}

    def load_checkpoint(self, checkpoint):
        self.log_name = checkpoint['log_name']
        self.ckpt_dir = checkpoint['ckpt_dir']
        self.round = checkpoint['round']
        self.discovered_samples = checkpoint['discovered_samples']
        self.discovered_classes = checkpoint['discovered_classes']
        self.open_classes = checkpoint['open_classes']
        self.acc_results_list = checkpoint['acc_results_list']

    def init_round(self, discovered_samples, open_examples, discovered_classes, open_classes):
        """ Initialize 
        """
        self.discovered_samples = discovered_samples
        self.open_examples = open_examples
        self.discovered_classes = discovered_classes
        self.open_classes = open_classes

    def log_round(self, round_i, discovered_samples, discovered_classes, acc_results):
        self.round = round_i
        self.discovered_samples = discovered_samples
        self.discovered_classes = discovered_classes
        self.acc_results_list.append(acc_results)

    def finish(self):
        pass
        