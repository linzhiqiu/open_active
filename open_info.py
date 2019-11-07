import torch
import torch.nn.functional as F
from tqdm import tqdm

class OpenCollector(object):
    def __init__(self, trainer_machine):
        '''Args:
            trainer_machine
        '''
        super(OpenCollector, self).__init__()
        self.trainer_machine = trainer_machine

    def gather_open_info(self, dataloader, device='cuda'):
        ''' Return the result of running model on dataloader
            Args:
                dataloader : Cannot use shuffle=True or sampler. Assume same order as the dataset.
            Returns:
                scores : A 1-D tensor representing the open set scores of all instances. Higher means more like open set
        '''
        open_score_func = self.trainer_machine.get_open_score_func() #May register a handle object
        pbar = tqdm(dataloader, ncols=80)
        # Score each examples in the unlabeled pool
        scores = torch.Tensor().to(device)
        with torch.no_grad():
            for batch, data in enumerate(pbar):
                inputs, labels = data
                
                inputs = inputs.to(device)

                scores_batch_i = open_score_func(inputs).to(device)
                scores = torch.cat((scores,scores_batch_i))
        
        self.trainer_machine.remove_handle()
        return scores

