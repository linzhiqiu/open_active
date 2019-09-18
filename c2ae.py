import argparse, os, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils as vutils
from torch.optim import lr_scheduler
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from collections import OrderedDict
from tqdm import tqdm
import copy

from trainer_machine import Network, train_epochs
from instance_info import C2AEInfoCollector
import models
from utils import get_subset_dataloaders, get_subset_loader, get_loader, SetPrintMode, get_target_mapping_func, get_target_unmapping_dict
from global_setting import OPEN_CLASS_INDEX, UNSEEN_CLASS_INDEX, PRETRAINED_MODEL_PATH
# from vector import clamp_to_unit_sphere
SAVE_FIG_EVERY_EPOCH = 500
def train_autoencoder(autoencoder, dataloaders, optimizer_decoder, scheduler_decoder, alpha, num_classes, train_mode='default', device="cuda", start_epoch=0, max_epochs=50, verbose=True, save_output=False):
    assert start_epoch < max_epochs
    avg_loss = 0.
    avg_match_loss = 0.
    avg_nonmatch_loss = 0.

    assert train_mode in ['default', 'a_minus_1', 'default_mse', 'a_minus_1_mse', 'default_bce', 'a_minus_1_bce']
    if train_mode == 'default':
        nonmatch_ratio = 1.-alpha
        criterion = nn.L1Loss()
    elif train_mode == 'a_minus_1':
        nonmatch_ratio = alpha-1.
        criterion = nn.L1Loss()
    elif train_mode == 'default_mse':
        nonmatch_ratio = 1.-alpha
        criterion = nn.MSELoss()
    elif train_mode == 'a_minus_1_mse':
        nonmatch_ratio = alpha-1.
        criterion = nn.MSELoss()
    elif train_mode == 'default_bce':
        nonmatch_ratio = 1.-alpha
        criterion = nn.BCELoss()
    elif train_mode == 'a_minus_1_bce':
        nonmatch_ratio = alpha-1.
        criterion = nn.BCELoss()

    for epoch in range(start_epoch, max_epochs):
        for phase in dataloaders.keys():
            if phase == "train":
                scheduler_decoder.step()
                autoencoder.train()
            else:
                autoencoder.eval()

            running_match_loss = 0.0
            running_nonmatch_loss = 0.0
            count = 0

            if verbose:
                pbar = tqdm(dataloaders[phase], ncols=80)
            else:
                pbar = dataloaders[phase]

            for batch, data in enumerate(pbar):
                inputs, labels = data
                count += inputs.size(0)
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer_decoder.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    matched_outputs = autoencoder(inputs, labels)
                    match_loss = criterion(matched_outputs, inputs)

                    dist = torch.zeros((inputs.shape[0], num_classes-1)).to(inputs.device) + (1/float(num_classes-1))
                    perm = torch.multinomial(dist,1).reshape(-1)
                    nm_labels = torch.remainder(labels + perm + 1, num_classes)
                    nonmatch_outputs = autoencoder(inputs, nm_labels)
                    nonmatch_loss = criterion(nonmatch_outputs, inputs)
                    
                    if save_output and (batch == len(dataloaders[phase])-1):
                        matched_results = vutils.make_grid(matched_outputs.detach().cpu(), padding=2, normalize=True)
                        nonmatch_results = vutils.make_grid(nonmatch_outputs.detach().cpu(), padding=2, normalize=True)

                        # if (batch % SAVE_FIG_EVERY_EPOCH == 0) and ((epoch == max_epochs-1) and (batch == len(dataloader)-1)):
                        
                        plt.figure(figsize=(25,25))
                        plt.subplot(1,3,1)
                        plt.axis("off")
                        plt.title("Original Images")
                        plt.imshow(np.transpose(vutils.make_grid(inputs, padding=2, normalize=True).cpu(),(1,2,0)))
                        
                        plt.subplot(1,3,2)
                        plt.axis("off")
                        plt.title("Match Reconstruction")
                        plt.imshow(np.transpose(matched_results,(1,2,0)))

                        plt.subplot(1,3,3)
                        plt.axis("off")
                        plt.title("NonMatch Reconstruction")
                        plt.imshow(np.transpose(nonmatch_results,(1,2,0)))

                        save_dir = f"c2ae_results/reconstruction_{train_mode}_a_{alpha}_epoch_{max_epochs}_phase_{phase}"
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        plt.savefig(save_dir + os.sep + str(epoch) + ".png")

                    loss = alpha * match_loss + nonmatch_ratio * nonmatch_loss
                    if phase == 'train':
                        loss.backward()
                        optimizer_decoder.step()

                    # statistics
                    running_match_loss += match_loss.item() * inputs.size(0)
                    running_nonmatch_loss += nonmatch_loss.item() * inputs.size(0)
                    if verbose:
                        pbar.set_postfix(loss=(running_match_loss+running_nonmatch_loss)/(count*2),
                                         match_loss=running_match_loss/count,
                                         nm_loss=running_nonmatch_loss/count,
                                         epoch=epoch,
                                         phase=phase)

                avg_loss = float(running_match_loss+running_nonmatch_loss)/(count*2)
                avg_match_loss = float(running_match_loss)/count
                avg_nonmatch_loss = float(running_nonmatch_loss)/count
    return avg_match_loss, avg_nonmatch_loss

def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class generator32(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, **kwargs):
        super(self.__class__, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 512*2*2, bias=False)

        self.conv2_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.ConvTranspose2d(   512,      512, 4, stride=2, padding=1, bias=False)
        self.conv3_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False)
        self.conv4_in = nn.ConvTranspose2d(latent_size, 256, 1, stride=1, padding=0, bias=False)
        self.conv4 = nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False)
        self.conv5 = nn.ConvTranspose2d(   128,        3, 4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)

        self.batch_size = batch_size
        self.apply(weights_init)

    def forward(self, x, input_scale=1):
        batch_size = x.shape[0]
        if input_scale <= 1:
            x = self.fc1(x)
            x = x.resize(batch_size, 512, 2, 2)

        # 512 x 2 x 2
        if input_scale == 2:
            x = x.view(batch_size, self.latent_size, 2, 2)
            x = self.conv2_in(x)
        if input_scale <= 2:
            x = self.conv2(x)
            x = nn.LeakyReLU()(x)
            x = self.bn2(x)

        # 512 x 4 x 4
        if input_scale == 4:
            x = x.view(batch_size, self.latent_size, 4, 4)
            x = self.conv3_in(x)
        if input_scale <= 4:
            x = self.conv3(x)
            x = nn.LeakyReLU()(x)
            x = self.bn3(x)

        # 256 x 8 x 8
        if input_scale == 8:
            x = x.view(batch_size, self.latent_size, 8, 8)
            x = self.conv4_in(x)
        if input_scale <= 8:
            x = self.conv4(x)
            x = nn.LeakyReLU()(x)
            x = self.bn4(x)
        # 128 x 16 x 16
        x = self.conv5(x)
        # 3 x 32 x 32
        # x = nn.Sigmoid()(x)
        x = nn.Tanh()(x) # Since we use zero mean normalization
        return x

class Decoder(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, num_classes=0):
        super(self.__class__, self).__init__()
        assert num_classes > 0
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.generator = generator32(latent_size=latent_size, batch_size=batch_size)
        self.h_y = torch.nn.Linear(num_classes, latent_size, bias=True)
        self.h_b = torch.nn.Linear(num_classes, latent_size, bias=True)

    def forward(self, x, labels):
        label_vectors = torch.zeros(x.shape[0], self.num_classes).to(x.device) - 1.
        label_vectors[torch.arange(x.shape[0]), labels] = 1.
        y = self.h_y(label_vectors)
        b = self.h_b(label_vectors)
        z = x*y + b
        out = self.generator(z)
        return out

class ResNetAutoEncoder(nn.Module):
    def __init__(self, resnet_model, decoder):
        super(self.__class__, self).__init__()
        self.encoder = resnet_model
        self.decoder = decoder

    def forward(self, x, labels):
        out = F.relu(self.encoder.bn1(self.encoder.conv1(x)))
        out = self.encoder.layer1(out)
        out = self.encoder.layer2(out)
        out = self.encoder.layer3(out)
        out = self.encoder.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.decoder(out, labels)
        return out


class C2AE(Network):
    def __init__(self, *args, **kwargs):
        super(C2AE, self).__init__(*args, **kwargs)
        self.c2ae_train_mode = self.config.c2ae_train_mode
        self.c2ae_alpha = self.config.c2ae_alpha

        self.model = self._get_network_model() # need to remove last relu layer
        
        # These network will be init in train() function
        self.autoencoder = None
        
        self.info_collector_class = C2AEInfoCollector

        if self.config.pseudo_open_set != None:
            raise NotImplementedError() # Disable pseudo open set training


    def _train(self, model, s_train, seen_classes, start_epoch=0):
        self._train_mode(model)
        target_mapping_func = self._get_target_mapp_func(seen_classes)
        self.dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                  list(s_train),
                                                  [], # TODO: Make a validation set
                                                  target_mapping_func,
                                                  batch_size=self.config.batch,
                                                  workers=self.config.workers)
        
        # Restore last fc layer
        self._update_last_layer(model, len(seen_classes), device=self.device)
        optimizer = self._get_network_optimizer(model)
        scheduler = self._get_network_scheduler(optimizer)

        self.criterion = self._get_criterion(self.dataloaders['train'],
                                             seen_classes=seen_classes,
                                             criterion_class=self.criterion_class)

        with SetPrintMode(hidden=not self.config.verbose):
            train_loss, train_acc = train_epochs(
                                        model,
                                        self.dataloaders,
                                        optimizer,
                                        scheduler,
                                        self.criterion,
                                        device=self.device,
                                        start_epoch=start_epoch,
                                        max_epochs=self.max_epochs,
                                        verbose=self.config.verbose,
                                    )
        print(f"Train => {self.round} round => "
              f"Loss {train_loss}, Accuracy {train_acc}")
        print(f"Finish training the encoder. Now train the decoder.")
        
        self.decoder = Decoder(latent_size=2048, batch_size=self.config.batch, num_classes=len(seen_classes)).to(self.device)
        optimizer_decoder = self._get_network_optimizer(self.decoder)
        scheduler_decoder = self._get_network_scheduler(optimizer_decoder)
        self.autoencoder = ResNetAutoEncoder(model, self.decoder)
        # criterion_autoencoder = torch.nn.L1Loss()
        # criterion_autoencoder = torch.nn.MSELoss()
        # criterion_autoencoder = torch.nn.BCELoss()
        with SetPrintMode(hidden=not self.config.verbose):
            match_loss, nm_loss =   train_autoencoder(
                                        self.autoencoder,
                                        self.dataloaders,
                                        optimizer_decoder,
                                        scheduler_decoder,
                                        self.c2ae_alpha,
                                        len(seen_classes),
                                        train_mode=self.c2ae_train_mode,
                                        device=self.device,
                                        start_epoch=0,
                                        # max_epochs=self.max_epochs,
                                        max_epochs=100,
                                        save_output=True,
                                        verbose=self.config.verbose,
                                    )
        print(f"Train => {self.round} round => "
              f"Match Loss {match_loss}, Non-match Loss {nm_loss}")
        return train_loss, train_acc

    def _get_open_set_pred_func(self):
        assert self.config.network_eval_mode in ['threshold', 'dynamic_threshold', 'pseuopen_threshold']
        if self.config.network_eval_mode == 'threshold':
            threshold = self.config.network_eval_threshold
        elif self.config.network_eval_mode in ['dynamic_threshold', 'pseuopen_threshold']:
            raise NotImplementedError()

        def open_set_prediction(outputs, inputs=None):
            num_classes = outputs.shape[1]
            k_labels = torch.arange(num_classes).unsqueeze(1).expand(-1, outputs.shape[0])

            for class_i in range(num_classes):
                errors = torch.abs(inputs - self.autoencoder(inputs, k_labels[class_i])).view(outputs.shape[0], -1).mean(1)
                if class_i == 0:
                    min_errors = errors
                else:
                    min_errors = torch.min(min_errors, errors)

            scores = min_errors

            softmax_outputs = F.softmax(outputs, dim=1)
            softmax_max, softmax_preds = torch.max(softmax_outputs, 1)

            assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
            self.thresholds_checkpoints[self.round]['open_set_score'] += (-scores).tolist()
            self.thresholds_checkpoints[self.round]['closed_predicted'] += softmax_preds.tolist()
            self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += softmax_max.tolist()
            self.thresholds_checkpoints[self.round]['open_predicted'] += softmax_preds.tolist()
            self.thresholds_checkpoints[self.round]['open_argmax_prob'] += softmax_max.tolist()
            
            preds = torch.where(scores < threshold,
                                torch.LongTensor([UNSEEN_CLASS_INDEX]).to(outputs.device), 
                                softmax_preds)
            return preds
        return open_set_prediction


    def _get_network_model(self):
        """ Get the regular softmax network model without last relu layer
        """
        model = getattr(models, self.config.arch)(last_relu=False)
        if self.config.pretrained != None:
            state_dict = self._get_pretrained_model_state_dict()
            model.load_state_dict(state_dict)
            del state_dict
        else:
            print("Using random initialized model")
        return model.to(self.device)
        