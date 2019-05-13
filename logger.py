

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
        self.mult_acc = []
        self.open_acc = []

    def get_checkpoint(self):
        return {'log_name' : self.log_name,
                'ckpt_dir' : self.ckpt_dir,
                'round' : self.round,
                's_train' : self.s_train,
                'seen_classes' : self.seen_classes,
                'multi_class_acc' : self.mult_acc,
                'open_set_acc' : self.open_acc}

    def load_checkpoint(self, checkpoint):
        self.log_name = checkpoint['log_name']
        self.ckpt_dir = checkpoint['ckpt_dir']
        self.round = checkpoint['round']
        self.s_train = checkpoint['s_train']
        self.seen_classes = checkpoint['seen_classes']
        self.multi_class_acc = checkpoint['multi_class_acc']
        self.open_set_acc = checkpoint['open_set_acc']

    def init_round(self, s_train, seen_classes):
        """ Initialize 
        """
        self.s_train = s_train
        self.seen_classes = seen_classes

    def log_round(self, round_i, s_train, seen_classes, multi_class_acc, open_set_acc):
        self.round = round_i
        self.s_train = s_train
        self.seen_classes = seen_classes
        self.mult_acc.append(multi_class_acc)
        self.open_acc.append(open_set_acc)

    def finish(self):
        pass
        