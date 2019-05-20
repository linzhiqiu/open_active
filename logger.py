

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
        self.s_train = None
        self.seen_classes = None
        self.acc_results_list = []

    def get_checkpoint(self):
        return {'log_name' : self.log_name,
                'ckpt_dir' : self.ckpt_dir,
                'round' : self.round,
                's_train' : self.s_train,
                'seen_classes' : self.seen_classes,
                'open_classes' : self.open_classes,
                'acc_results_list' : self.acc_results_list}

    def load_checkpoint(self, checkpoint):
        self.log_name = checkpoint['log_name']
        self.ckpt_dir = checkpoint['ckpt_dir']
        self.round = checkpoint['round']
        self.s_train = checkpoint['s_train']
        self.seen_classes = checkpoint['seen_classes']
        self.open_classes = checkpoint['open_classes']
        self.acc_results_list = checkpoint['acc_results_list']

    def init_round(self, s_train, open_examples, seen_classes, open_classes):
        """ Initialize 
        """
        self.s_train = s_train
        self.open_examples = open_examples
        self.seen_classes = seen_classes
        self.open_classes = open_classes

    def log_round(self, round_i, s_train, seen_classes, acc_results):
        self.round = round_i
        self.s_train = s_train
        self.seen_classes = seen_classes
        self.acc_results_list.append(acc_results)

    def finish(self):
        pass
        