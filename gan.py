import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from trainer_machine import Network
import models
from utils import get_subset_dataloaders
from global_setting import GAN_SETUP_DICT
from transform import get_dcgan_transform
from global_setting import OPEN_CLASS_INDEX, UNDISCOVERED_CLASS_INDEX

############
# Below are helper functions and classes for GAN training
############
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nc=3, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class FeatureGenerator(nn.Module):
    def __init__(self, nz=2048, nc=100, ngf=512):
        super(FeatureGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nc, ngf),
            nn.ReLU(),
            nn.Linear(ngf, ngf),
            nn.ReLU(),
            nn.Linear(ngf, nz),
        )

    def forward(self, input):
        return self.main(input)

class FeatureDiscriminator(nn.Module):
    def __init__(self, nz=2048, ndf=512):
        super(FeatureDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, ndf),
            nn.ReLU(),
            nn.Linear(ndf, ndf),
            nn.ReLU(),
            nn.Linear(ndf, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class GAN(Network):
    def __init__(self, *args, **kwargs):
        super(GAN, self).__init__(*args, **kwargs)
        assert self.config.pseudo_open_set == None # Pseudo open set disabled
        assert self.pseudo_open_set_rounds == 0

        self.gan_player = self.config.gan_player
        self.gan_mode = self.config.gan_mode
        self.gan_setup = self.config.gan_setup
        if self.config.save_gan_output:
            self.save_gan_output = os.path.join("gan_outputs", str(self.gan_player),
                str(self.gan_mode), str(self.gan_setup))
            print(f"Output saving to {self.save_gan_output}")
            if not os.path.exists(self.save_gan_output):
                os.makedirs(self.save_gan_output)
        else:
            self.save_gan_output = None

        self.setting = GAN_SETUP_DICT[self.gan_player][self.gan_mode][self.gan_setup]

    def _train(self, model, discovered_samples, discovered_classes, start_epoch=0):
        # Save these variable for self._get_open_set_pred_func(self)
        self.discovered_samples = discovered_samples
        self.discovered_classes = discovered_classes
        return super(GAN, self)._train(model, discovered_samples, discovered_classes, start_epoch=start_epoch)

    def _eval(self, model, test_dataset, discovered_classes, verbose=False, training_features=None):
        # assert hasattr(self, 'dataloaders') # Should be updated after calling super._train()
        # self.num_discovered_classes = len(discovered_classes)
        self._reset_open_set_stats() # Open set status is the summary of 1/ Number of threshold reject 2/ Number of Open Class reject
        eval_result = super(GAN, self)._eval(model, test_dataset, discovered_classes, verbose=verbose)
        if verbose:
            self._print_open_set_stats()
        return eval_result

    def _get_open_set_pred_func(self):
        assert hasattr(self, 'discovered_samples') and hasattr(self, 'discovered_classes')
        raise NotImplementedError()

    def _print_open_set_stats(self):
        # print(f"Rejection details: Total rejects {self.open_set_stats['total_reject']}. "
        #       f"By threshold ({self.osdn_eval_threshold}) {self.open_set_stats['threshold_reject']}. "
        #       f"By being open class {self.open_set_stats['open_class_reject']}. "
        #       f"By both {self.open_set_stats['both_reject']}. ")
        raise NotImplementedError()

    def _update_open_set_stats(self, gan_max, gan_preds):
        # For each batch
        # self.open_set_stats['threshold_reject'] += float(torch.sum((openmax_max < self.osdn_eval_threshold) & ~(openmax_preds == self.num_discovered_classes) ))
        # self.open_set_stats['open_class_reject'] += float(torch.sum(~(openmax_max < self.osdn_eval_threshold) & (openmax_preds == self.num_discovered_classes) ))
        # self.open_set_stats['both_reject'] += float(torch.sum((openmax_max < self.osdn_eval_threshold) & (openmax_preds == self.num_discovered_classes) ))
        # self.open_set_stats['total_reject'] += float(torch.sum((openmax_max < self.osdn_eval_threshold) | (openmax_preds == self.num_discovered_classes) ))
        # assert self.open_set_stats['threshold_reject'] + self.open_set_stats['open_class_reject'] + self.open_set_stats['both_reject'] == self.open_set_stats['total_reject']
        raise NotImplementedError()

    def _reset_open_set_stats(self):
        # threshold_reject and open_class_reject are mutually exclusive
        # self.gan_stats = {'threshold_reject': 0., 
        #                   'open_class_reject': 0.,
        #                   'both_reject': 0.,
        #                   'total_reject': 0.}
        raise NotImplementedError()

class FeatureGAN(GAN):
    def __init__(self, *args, **kwargs):
        super(FeatureGAN, self).__init__(*args, **kwargs)

    def _get_network_model(self):
        """ Get the regular softmax network model
        """
        model = getattr(models, self.config.arch)(last_relu=False)
        if self.config.pretrained != None:
            state_dict = self._get_pretrained_model_state_dict()
            model.load_state_dict(state_dict)
            del state_dict
        else:
            print("Using random initialized model")
        return model.to(self.device)

    def _eval(self, model, test_dataset, discovered_classes, verbose=False, training_features=None):
        eval_result = super(FeatureGAN, self)._eval(model, test_dataset, discovered_classes, verbose=verbose)
        if not hasattr(self.model, 'forward_hook_handle'):
            raise NotImplementedError()
        else:
            self.model.forward_hook_handle.remove()
        return eval_result

class GANFactory(object):
    def __init__(self, config):
        self.config = config
        self.gan_player = self.config.gan_player
        self.gan_mode = self.config.gan_mode

    def gan_class(self):
        if self.gan_player == 'single':
            if self.gan_mode == 'ImageLevelGAN':
                return SingleImageLevelGAN
            if self.gan_mode == 'FeatureLevelGAN':
                return SingleFeatureLevelGAN
        elif self.gan_player in ['multiple']:
            if self.gan_mode == 'ImageLevelGAN':
                return MultipleImageLevelGAN
            if self.gan_mode == 'FeatureLevelGAN':
                return MultipleFeatureLevelGAN
        elif self.gan_player in ['background', 'background_noise']:
            if self.gan_mode == 'ImageLevelGAN':
                return BackgroundImageGAN
            if self.gan_mode == 'FeatureLevelGAN':
                return BackgroundFeatureGAN
        # else:
        raise NotImplementedError()


class SingleImageLevelGAN(GAN):
    def __init__(self, *args, **kwargs):
        super(SingleImageLevelGAN, self).__init__(*args, **kwargs)

    def _get_open_set_pred_func(self):
        assert hasattr(self, 'discovered_samples') and hasattr(self, 'discovered_classes')
        # ## Change transformation of dataset for gan training
        # old_transform = self.train_instance.train_dataset.transform
        # self.train_instance.train_dataset.transform = get_dcgan_transform()
        gan_dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                 list(self.discovered_samples),
                                                 [], # TODO: Make a validation set
                                                 None, # Single ImageLevel Gan doesn't require class label
                                                 batch_size=self.config.batch,
                                                 workers=self.config.workers)
        
        netG, netD = train_dcgan(gan_dataloaders['train'],
                                 self.setting,
                                 device=self.config.device,
                                 save_gan_output=self.save_gan_output)

        # self.train_instance.train_dataset.transform = old_transform

        def open_set_prediction(outputs, inputs=None):
            d_outputs = netD(inputs).squeeze()
            softmax_outputs = F.softmax(outputs, dim=1)
            _, softmax_preds = torch.max(softmax_outputs, 1)
            preds = torch.where(d_outputs < 0.5,
                                torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
                                softmax_preds)
            return preds
        return open_set_prediction


    def _print_open_set_stats(self):
        pass

    def _update_open_set_stats(self, gan_max, gan_preds):
        pass

    def _reset_open_set_stats(self):
        pass

class SingleImageLevelGAN(GAN):
    def __init__(self, *args, **kwargs):
        super(SingleImageLevelGAN, self).__init__(*args, **kwargs)

    def _get_open_set_pred_func(self):
        assert hasattr(self, 'discovered_samples') and hasattr(self, 'discovered_classes')
        # ## Change transformation of dataset for gan training
        # old_transform = self.train_instance.train_dataset.transform
        # self.train_instance.train_dataset.transform = get_dcgan_transform()
        gan_dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                 list(self.discovered_samples),
                                                 [], # TODO: Make a validation set
                                                 None, # Single ImageLevel Gan doesn't require class label
                                                 batch_size=self.config.batch,
                                                 workers=self.config.workers)
        
        netG, netD = train_dcgan(gan_dataloaders['train'],
                                 self.setting,
                                 device=self.config.device,
                                 save_gan_output=self.save_gan_output)

        # self.train_instance.train_dataset.transform = old_transform

        def open_set_prediction(outputs, inputs=None):
            d_outputs = netD(inputs).squeeze()
            softmax_outputs = F.softmax(outputs, dim=1)
            _, softmax_preds = torch.max(softmax_outputs, 1)
            preds = torch.where(d_outputs < 0.5,
                                torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
                                softmax_preds)
            return preds
        return open_set_prediction


    def _print_open_set_stats(self):
        pass

    def _update_open_set_stats(self, gan_max, gan_preds):
        pass

    def _reset_open_set_stats(self):
        pass


class SingleFeatureLevelGAN(FeatureGAN):
    def __init__(self, *args, **kwargs):
        super(SingleFeatureLevelGAN, self).__init__(*args, **kwargs)

    def _get_open_set_pred_func(self):
        assert hasattr(self, 'discovered_samples') and hasattr(self, 'discovered_classes')

        self.model.feature_vectors = []
        def hook_input(module, input, output):
            self.model.feature_vectors.append(input[0])

        self.model.forward_hook_handle = self.model.fc.register_forward_hook(hook_input)

        gan_dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                 list(self.discovered_samples),
                                                 [], # TODO: Make a validation set
                                                 None, # Single ImageLevel Gan doesn't require class label
                                                 batch_size=self.config.batch,
                                                 workers=self.config.workers)
        
        netG, netD = train_feature_level_dcgan(gan_dataloaders['train'],
                                               self.setting,
                                               model=self.model,
                                               device=self.config.device)

        def open_set_prediction(outputs, inputs=None):
            self.model.feature_vectors = []
            _ = self.model(inputs)
            d_outputs = netD(self.model.feature_vectors[0]).squeeze()
            softmax_outputs = F.softmax(outputs, dim=1)
            _, softmax_preds = torch.max(softmax_outputs, 1)
            preds = torch.where(d_outputs < 0.5,
                                torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
                                softmax_preds)
            return preds
        return open_set_prediction


    def _print_open_set_stats(self):
        pass

    def _update_open_set_stats(self, gan_max, gan_preds):
        pass

    def _reset_open_set_stats(self):
        pass

class MultipleImageLevelGAN(GAN):
    def __init__(self, *args, **kwargs):
        super(MultipleImageLevelGAN, self).__init__(*args, **kwargs)
        self.gan_multi = self.config.gan_multi

    def _get_open_set_pred_func(self):
        assert hasattr(self, 'discovered_samples') and hasattr(self, 'discovered_classes')
        gan_list = {discovered_class_index : {'netG': None, 
                                        'netD' : None,
                                        'samples' : list(filter(lambda x: self.train_instance.train_labels[x] == discovered_class_index, self.discovered_samples))} 
                    for discovered_class_index in self.discovered_classes}
        for discovered_class_index in gan_list.keys():  
            gan_dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                     list(gan_list[discovered_class_index]['samples']),
                                                     [], # TODO: Make a validation set
                                                     None, # Multiple ImageLevel Gan doesn't require class label
                                                     batch_size=self.config.batch,
                                                     workers=self.config.workers)
            if self.save_gan_output:
                save_gan_output_class_dir = self.save_gan_output+os.sep+"class_"+str(discovered_class_index)
                if not os.path.exists(save_gan_output_class_dir):
                    os.makedirs(save_gan_output_class_dir)
            else:
                save_gan_output_class_dir = None
            netG, netD = train_dcgan(gan_dataloaders['train'],
                                     self.setting,
                                     device=self.config.device,
                                     save_gan_output=save_gan_output_class_dir)
            gan_list[discovered_class_index]['netD'] = netD
            gan_list[discovered_class_index]['netG'] = netG

        target_mapping_func = self._get_target_mapp_func(self.discovered_classes)

        def open_set_prediction(outputs, inputs=None):
            d_preds = torch.ones(outputs.shape[0]).byte().to(inputs.device)
            softmax_outputs = F.softmax(outputs, dim=1)
            _, softmax_preds = torch.max(softmax_outputs, 1)
            _, softmax_worst = torch.min(softmax_outputs, 1)

            for i in gan_list.keys():
                if self.gan_multi == 'all':
                    d_preds = d_preds & (gan_list[i]['netD'](inputs).squeeze() < 0.5)
                elif self.gan_multi == "highest":
                    d_preds = torch.where(softmax_preds == target_mapping_func(i),
                                          d_preds & (gan_list[i]['netD'](inputs).squeeze() < 0.5),
                                          d_preds)
                elif self.gan_multi == "lowest":
                    d_preds = torch.where(softmax_worst == target_mapping_func(i),
                                          d_preds & (gan_list[i]['netD'](inputs).squeeze() < 0.5),
                                          d_preds)
            
            preds = torch.where(d_preds,
                                torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
                                softmax_preds)
            return preds
        return open_set_prediction


    def _print_open_set_stats(self):
        pass

    def _update_open_set_stats(self, gan_max, gan_preds):
        pass

    def _reset_open_set_stats(self):
        pass

class BackgroundImageGAN(GAN):
    def __init__(self, *args, **kwargs):
        super(BackgroundImageGAN, self).__init__(*args, **kwargs)
        self.gan_multi = self.config.gan_multi
        self.use_noise = self.gan_player == 'background_noise'

    def _get_open_set_pred_func(self):
        assert hasattr(self, 'discovered_samples') and hasattr(self, 'discovered_classes')
        gan_list = {discovered_class_index : {'netG': None, 
                                              'netD' : None,
                                              'samples' : list(filter(lambda x: self.train_instance.train_labels[x] == discovered_class_index, self.discovered_samples)),
                                              'backgrounds' : list(filter(lambda x: self.train_instance.train_labels[x] != discovered_class_index, self.discovered_samples))} 
                    for discovered_class_index in self.discovered_classes}
        for discovered_class_index in gan_list.keys():  
            gan_dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                     list(gan_list[discovered_class_index]['samples']),
                                                     [], # TODO: Make a validation set
                                                     None, # Multiple ImageLevel Gan doesn't require class label
                                                     batch_size=self.config.batch,
                                                     workers=self.config.workers)
            background_dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                            list(gan_list[discovered_class_index]['backgrounds']),
                                                            [], # TODO: Make a validation set
                                                            None, # Multiple ImageLevel Gan doesn't require class label
                                                            batch_size=self.config.batch,
                                                            workers=self.config.workers)
            if self.save_gan_output:
                save_gan_output_class_dir = self.save_gan_output+os.sep+"class_"+str(discovered_class_index)
                if not os.path.exists(save_gan_output_class_dir):
                    os.makedirs(save_gan_output_class_dir)
            else:
                save_gan_output_class_dir = None
            netG, netD = train_dcgan_with_background(gan_dataloaders['train'],
                                                     background_dataloaders['train'],
                                                     self.setting,
                                                     device=self.config.device,
                                                     use_noise=self.use_noise,
                                                     save_gan_output=save_gan_output_class_dir)
            gan_list[discovered_class_index]['netD'] = netD
            gan_list[discovered_class_index]['netG'] = netG

        target_mapping_func = self._get_target_mapp_func(self.discovered_classes)

        def open_set_prediction(outputs, inputs=None):
            d_preds = torch.ones(outputs.shape[0]).byte().to(inputs.device)
            softmax_outputs = F.softmax(outputs, dim=1)
            _, softmax_preds = torch.max(softmax_outputs, 1)
            _, softmax_worst = torch.min(softmax_outputs, 1)

            for i in gan_list.keys():
                if self.gan_multi == 'all':
                    d_preds = d_preds & (gan_list[i]['netD'](inputs).squeeze() < 0.5)
                elif self.gan_multi == "highest":
                    d_preds = torch.where(softmax_preds == target_mapping_func(i),
                                          d_preds & (gan_list[i]['netD'](inputs).squeeze() < 0.5),
                                          d_preds)
                elif self.gan_multi == "lowest":
                    d_preds = torch.where(softmax_worst == target_mapping_func(i),
                                          d_preds & (gan_list[i]['netD'](inputs).squeeze() < 0.5),
                                          d_preds)
            
            preds = torch.where(d_preds,
                                torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
                                softmax_preds)
            return preds
        return open_set_prediction


    def _print_open_set_stats(self):
        pass

    def _update_open_set_stats(self, gan_max, gan_preds):
        pass

    def _reset_open_set_stats(self):
        pass


class MultipleFeatureLevelGAN(GAN):
    def __init__(self, *args, **kwargs):
        super(MultipleFeatureLevelGAN, self).__init__(*args, **kwargs)
        self.gan_multi = self.config.gan_multi

    def _get_open_set_pred_func(self):
        assert hasattr(self, 'discovered_samples') and hasattr(self, 'discovered_classes')

        self.model.feature_vectors = []
        def hook_input(module, input, output):
            self.model.feature_vectors.append(input[0])

        self.model.forward_hook_handle = self.model.fc.register_forward_hook(hook_input)

        gan_list = {discovered_class_index : {'netG': None, 
                                        'netD' : None,
                                        'samples' : list(filter(lambda x: self.train_instance.train_labels[x] == discovered_class_index, self.discovered_samples))} 
                    for discovered_class_index in self.discovered_classes}
        for discovered_class_index in gan_list.keys():  
            gan_dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                     list(gan_list[discovered_class_index]['samples']),
                                                     [], # TODO: Make a validation set
                                                     None, # Multiple ImageLevel Gan doesn't require class label
                                                     batch_size=self.config.batch,
                                                     workers=self.config.workers)
            
            netG, netD = train_feature_level_dcgan(gan_dataloaders['train'],
                                                   self.setting,
                                                   model=self.model,
                                                   device=self.config.device)
            gan_list[discovered_class_index]['netD'] = netD
            gan_list[discovered_class_index]['netG'] = netG

        target_mapping_func = self._get_target_mapp_func(self.discovered_classes)

        def open_set_prediction(outputs, inputs=None):
            d_preds = torch.ones(outputs.shape[0]).byte().to(inputs.device)
            softmax_outputs = F.softmax(outputs, dim=1)
            _, softmax_preds = torch.max(softmax_outputs, 1)
            _, softmax_worst = torch.min(softmax_outputs, 1)

            self.model.feature_vectors = []
            _ = self.model(inputs) # TO gather the feature vector

            for i in gan_list.keys():
                if self.gan_multi == 'all':
                    d_preds = d_preds & (gan_list[i]['netD'](self.model.feature_vectors[0]).squeeze() < 0.5)
                elif self.gan_multi == "highest":
                    d_preds = torch.where(softmax_preds == target_mapping_func(i),
                                          d_preds & (gan_list[i]['netD'](self.model.feature_vectors[0]).squeeze() < 0.5),
                                          d_preds)
                elif self.gan_multi == "lowest":
                    d_preds = torch.where(softmax_worst == target_mapping_func(i),
                                          d_preds & (gan_list[i]['netD'](self.model.feature_vectors[0]).squeeze() < 0.5),
                                          d_preds)
            
            preds = torch.where(d_preds,
                                torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
                                softmax_preds)
            return preds
        return open_set_prediction


    def _print_open_set_stats(self):
        pass

    def _update_open_set_stats(self, gan_max, gan_preds):
        pass

    def _reset_open_set_stats(self):
        pass

class BackgroundFeatureGAN(GAN):
    def __init__(self, *args, **kwargs):
        super(BackgroundFeatureGAN, self).__init__(*args, **kwargs)
        self.gan_multi = self.config.gan_multi
        self.use_noise = self.gan_player == 'background_noise'

    def _get_open_set_pred_func(self):
        assert hasattr(self, 'discovered_samples') and hasattr(self, 'discovered_classes')

        self.model.feature_vectors = []
        def hook_input(module, input, output):
            self.model.feature_vectors.append(input[0])

        self.model.forward_hook_handle = self.model.fc.register_forward_hook(hook_input)

        gan_list = {discovered_class_index : {'netG': None, 
                                        'netD' : None,
                                        'samples' : list(filter(lambda x: self.train_instance.train_labels[x] == discovered_class_index, self.discovered_samples)),
                                        'backgrounds' : list(filter(lambda x: self.train_instance.train_labels[x] != discovered_class_index, self.discovered_samples))} 
                    for discovered_class_index in self.discovered_classes}
        for discovered_class_index in gan_list.keys():  
            gan_dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                     list(gan_list[discovered_class_index]['samples']),
                                                     [], # TODO: Make a validation set
                                                     None, # Multiple ImageLevel Gan doesn't require class label
                                                     batch_size=self.config.batch,
                                                     workers=self.config.workers)
            background_dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                            list(gan_list[discovered_class_index]['backgrounds']),
                                                            [], # TODO: Make a validation set
                                                            None, # Multiple ImageLevel Gan doesn't require class label
                                                            batch_size=self.config.batch,
                                                            workers=self.config.workers)
            netG, netD = train_feature_level_dcgan_with_background(gan_dataloaders['train'],
                                                                   background_dataloaders['train'],
                                                                   self.setting,
                                                                   model=self.model,
                                                                   use_noise=self.use_noise,
                                                                   device=self.config.device)
            gan_list[discovered_class_index]['netD'] = netD
            gan_list[discovered_class_index]['netG'] = netG

        target_mapping_func = self._get_target_mapp_func(self.discovered_classes)

        def open_set_prediction(outputs, inputs=None):
            d_preds = torch.ones(outputs.shape[0]).byte().to(inputs.device)
            softmax_outputs = F.softmax(outputs, dim=1)
            _, softmax_preds = torch.max(softmax_outputs, 1)
            _, softmax_worst = torch.min(softmax_outputs, 1)

            self.model.feature_vectors = []
            _ = self.model(inputs) # TO gather the feature vector

            for i in gan_list.keys():
                if self.gan_multi == 'all':
                    d_preds = d_preds & (gan_list[i]['netD'](self.model.feature_vectors[0]).squeeze() < 0.5)
                elif self.gan_multi == "highest":
                    d_preds = torch.where(softmax_preds == target_mapping_func(i),
                                          d_preds & (gan_list[i]['netD'](self.model.feature_vectors[0]).squeeze() < 0.5),
                                          d_preds)
                elif self.gan_multi == "lowest":
                    d_preds = torch.where(softmax_worst == target_mapping_func(i),
                                          d_preds & (gan_list[i]['netD'](self.model.feature_vectors[0]).squeeze() < 0.5),
                                          d_preds)
            
            preds = torch.where(d_preds,
                                torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
                                softmax_preds)
            return preds
        return open_set_prediction


    def _print_open_set_stats(self):
        pass

    def _update_open_set_stats(self, gan_max, gan_preds):
        pass

    def _reset_open_set_stats(self):
        pass


def train_dcgan(dataloader, setting, device='cuda', save_gan_output=None):
    netG = Generator(nc=setting['nc'], nz=setting['nz'], ngf=setting['ngf']).to(device)
    netG.apply(weights_init)
    netD = Discriminator(nc=setting['nc'], ndf=setting['ndf']).to(device)
    netD.apply(weights_init)

    optimizerD = getattr(torch.optim, setting['optim'])(netD.parameters(), lr=setting['lr'], betas=(setting['beta1'], 0.999))
    optimizerG = getattr(torch.optim, setting['optim'])(netG.parameters(), lr=setting['lr'], betas=(setting['beta1'], 0.999))

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, setting['nz'], 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0


    # # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


    # print("Starting Training Loop...")
    # For each epoch
    for epoch in range(setting['num_epochs']):
        # For each batch in the dataloader
        accD_real = 0.
        accD_fake = 0.
        total_real = 0.
        total_fake = 0.
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            total_real += output.shape[0]
            accD_real += int((output >= 0.5).sum())
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, setting['nz'], 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            accD_fake += int((output < 0.5).sum())
            total_fake += output.shape[0]
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, setting['num_epochs'], i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == setting['num_epochs']-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1
        print(f"Epoch {epoch}: Correct on Real {accD_real}/{total_real}, on Fake {accD_fake}/{total_fake}.")
        if save_gan_output:
            # Grab a batch of real images from the dataloader
            real_batch = data
            # Plot the real images
            plt.figure(figsize=(15,15))
            plt.subplot(1,2,1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

            # Plot the fake images from the last epoch
            plt.subplot(1,2,2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list[-1],(1,2,0)))
            plt.savefig(save_gan_output + os.sep + str(epoch) + ".png")
    return netG, netD
    # plt.figure(figsize=(10,5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses,label="G")
    # plt.plot(D_losses,label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()

    # # Grab a batch of real images from the dataloader
    # real_batch = next(iter(dataloader))

    # # Plot the real images
    # plt.figure(figsize=(15,15))
    # plt.subplot(1,2,1)
    # plt.axis("off")
    # plt.title("Real Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # # Plot the fake images from the last epoch
    # plt.subplot(1,2,2)
    # plt.axis("off")
    # plt.title("Fake Images")
    # plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    # plt.show()

def train_feature_level_dcgan(dataloader, setting, model=None, device='cuda'):
    assert model != None and hasattr(model, 'feature_vectors') and hasattr(model, 'forward_hook_handle')
    netG = FeatureGenerator(nc=setting['nc'], nz=setting['nz'], ngf=setting['ngf']).to(device)
    netG.apply(weights_init)
    netD = FeatureDiscriminator(nz=setting['nz'], ndf=setting['ndf']).to(device)
    netD.apply(weights_init)

    optimizerD = getattr(torch.optim, setting['optim'])(netD.parameters(), lr=setting['lr'], betas=(setting['beta1'], 0.999))
    optimizerG = getattr(torch.optim, setting['optim'])(netG.parameters(), lr=setting['lr'], betas=(setting['beta1'], 0.999))

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, setting['nc'], device=device)
    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0


    # # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


    # print("Starting Training Loop...")
    # For each epoch
    for epoch in range(setting['num_epochs']):
        # For each batch in the dataloader
        accD_real = 0.
        accD_fake = 0.
        total_real = 0.
        total_fake = 0.
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            model.feature_vectors = []
            _ = model(data[0].to(device))
            real_cpu = model.feature_vectors[0]
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            total_real += output.shape[0]
            accD_real += int((output >= 0.5).sum())
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, setting['nc'], device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            accD_fake += int((output < 0.5).sum())
            total_fake += output.shape[0]
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, setting['num_epochs'], i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == setting['num_epochs']-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        print(f"Epoch {epoch}: Correct on Real {accD_real}/{total_real}, on Fake {accD_fake}/{total_fake}.")
    return netG, netD

def train_dcgan_with_background(dataloader, background_dataloader, setting, device='cuda', use_noise=False, save_gan_output=None):
    if use_noise:
        netG = Generator(nc=setting['nc'], nz=setting['nz'], ngf=setting['ngf']).to(device)
        netG.apply(weights_init)
        optimizerG = getattr(torch.optim, setting['optim'])(netG.parameters(), lr=setting['lr'], betas=(setting['beta1'], 0.999))
        img_list = []
        G_losses = []
        fixed_noise = torch.randn(64, setting['nz'], 1, 1, device=device)
    else:
        netG = None

    netD = Discriminator(nc=setting['nc'], ndf=setting['ndf']).to(device)
    netD.apply(weights_init)
    optimizerD = getattr(torch.optim, setting['optim'])(netD.parameters(), lr=setting['lr'], betas=(setting['beta1'], 0.999))
    

    criterion = nn.BCELoss()
    
    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    D_losses = []
    iters = 0


    # print("Starting Training Loop...")
    # For each epoch
    for epoch in range(setting['num_epochs']):
        # For each batch in the dataloader
        accD_real = 0.
        accD_back = 0.
        accD_fake = 0.
        total_real = 0.
        total_fake = 0.
        total_back = 0.
        background_iter = iter(background_dataloader)
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            total_real += output.shape[0]
            accD_real += int((output >= 0.5).sum())
            # Calculate gradients for D in backward pass
            errD_real.backward()
            errD = errD_real
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            if use_noise: 
                noise = torch.randn(b_size, setting['nz'], 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                accD_fake += int((output < 0.5).sum())
                total_fake += output.shape[0]
                # Calculate the gradients for this batch
                errD_fake.backward()
                errD += errD_fake
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches

                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=device)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                total_real += output.shape[0]
                accD_real += int((output >= 0.5).sum())
                # Calculate gradients for D in backward pass
                errD_real.backward()
                errD += errD_real
                D_x = output.mean().item()

            back_data = next(background_iter)[0].to(device)
            b_size = back_data.size(0)
            label = torch.full((b_size,), fake_label, device=device)
            output = netD(back_data).view(-1)
            errD_back = criterion(output, label)
            total_back += output.shape[0]
            accD_back += int((output < 0.5).sum())
            # Calculate gradients for D in backward pass
            errD_back.backward()
            D_b = output.mean().item()
            errD += errD_back
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if use_noise:
                netG.zero_grad()
                label = torch.full((fake.size(0),), real_label, device=device)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()
                G_losses.append(errG.item())

            # Output training stats
            if i % 50 == 0:
                if use_noise:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(b): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, setting['num_epochs'], i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_b, D_G_z1, D_G_z2))
                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (iters % 500 == 0) or ((epoch == setting['num_epochs']-1) and (i == len(dataloader)-1)):
                        with torch.no_grad():
                            fake = netG(fixed_noise).detach().cpu()
                        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                else:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tD(x): %.4f\tD(b): %.4f\t'
                          % (epoch, setting['num_epochs'], i, len(dataloader),
                             errD.item(), D_x, D_b))
            # Save Losses for plotting later
            D_losses.append(errD.item())

            iters += 1
        print(f"Epoch {epoch}: Correct on Real {accD_real}/{total_real}, on Fake {accD_fake}/{total_fake}, on Background {accD_back}/{total_back}.")
        if save_gan_output and use_noise:
            # Grab a batch of real images from the dataloader
            real_batch = data
            # Plot the real images
            plt.figure(figsize=(15,15))
            plt.subplot(1,2,1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

            # Plot the fake images from the last epoch
            plt.subplot(1,2,2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list[-1],(1,2,0)))
            plt.savefig(save_gan_output + os.sep + str(epoch) + ".png")
    return netG, netD

def train_feature_level_dcgan_with_background(dataloader, background_dataloader, setting, model=None, device='cuda', use_noise=False):
    assert model != None and hasattr(model, 'feature_vectors') and hasattr(model, 'forward_hook_handle')
    if use_noise:
        netG = FeatureGenerator(nc=setting['nc'], nz=setting['nz'], ngf=setting['ngf']).to(device)
        netG.apply(weights_init)
        optimizerG = getattr(torch.optim, setting['optim'])(netG.parameters(), lr=setting['lr'], betas=(setting['beta1'], 0.999))
        fixed_noise = torch.randn(64, setting['nc'], device=device)
        # Lists to keep track of progress
        img_list = []
        G_losses = []
    else:
        netG = None

    netD = FeatureDiscriminator(nz=setting['nz'], ndf=setting['ndf']).to(device)
    netD.apply(weights_init)
    optimizerD = getattr(torch.optim, setting['optim'])(netD.parameters(), lr=setting['lr'], betas=(setting['beta1'], 0.999))

    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0

    D_losses = []
    iters = 0


    # # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


    # print("Starting Training Loop...")
    # For each epoch
    for epoch in range(setting['num_epochs']):
        # For each batch in the dataloader
        accD_real = 0.
        accD_back = 0.
        accD_fake = 0.
        total_real = 0.
        total_fake = 0.
        total_back = 0.
        background_iter = iter(background_dataloader)
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            model.feature_vectors = []
            _ = model(data[0].to(device))
            real_cpu = model.feature_vectors[0]
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            total_real += output.shape[0]
            accD_real += int((output >= 0.5).sum())
            # Calculate gradients for D in backward pass
            errD_real.backward()
            errD = errD_real
            D_x = output.mean().item()


            ## Train with all-fake batch
            # Generate batch of latent vectors
            if use_noise: 
                noise = torch.randn(b_size, setting['nc'], device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                accD_fake += int((output < 0.5).sum())
                total_fake += output.shape[0]
                # Calculate the gradients for this batch
                errD_fake.backward()
                errD += errD_fake
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches

            back_data = next(background_iter)[0].to(device)
            b_size = back_data.size(0)
            # Format batch
            model.feature_vectors = []
            _ = model(back_data)
            back_feature = model.feature_vectors[0]
            label = torch.full((b_size,), fake_label, device=device)
            output = netD(back_feature).view(-1)
            errD_back = criterion(output, label)
            total_back += output.shape[0]
            accD_back += int((output < 0.5).sum())
            # Calculate gradients for D in backward pass
            errD_back.backward()
            D_b = output.mean().item()
            errD += errD_back
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if use_noise:
                netG.zero_grad()
                label = torch.full((fake.shape[0],), real_label, device=device) # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                if use_noise:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(b): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, setting['num_epochs'], i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_b, D_G_z1, D_G_z2))
                    G_losses.append(errG.item())
                else:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tD(x): %.4f\tD(b): %.4f\t'
                          % (epoch, setting['num_epochs'], i, len(dataloader),
                             errD.item(), D_x, D_b))

            # Save Losses for plotting later
            D_losses.append(errD.item())


            iters += 1
        print(f"Epoch {epoch}: Correct on Real {accD_real}/{total_real}, on Fake {accD_fake}/{total_fake}, on Background {accD_back}/{total_back}.")
    return netG, netD

if __name__ == '__main__':
    y = torch.randn(3,100)
    print(FeatureGenerator()(y).shape)
    y = torch.randn(3,2048)
    print(FeatureDiscriminator()(y).shape)

