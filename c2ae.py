import torch, torchvision
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os

from utils import get_subset_dataloaders, SetPrintMode
from global_setting import UNDISCOVERED_CLASS_INDEX

class Decoder(nn.Module):
    def __init__(self, latent_size=100, batch_size=64, num_classes=0, generator_class=None, **decoder_params):
        super(self.__class__, self).__init__()
        assert num_classes > 0
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.generator = generator_class(latent_size=latent_size, batch_size=batch_size, **decoder_params)
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

def weights_init(m):
    classname = m.__class__.__name__
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
        self.conv2 = nn.ConvTranspose2d(512,512, 4, stride=2, padding=1, bias=False)
        self.conv3_in = nn.ConvTranspose2d(latent_size, 512, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.ConvTranspose2d(512,256, 4, stride=2, padding=1, bias=False)
        self.conv4_in = nn.ConvTranspose2d(latent_size, 256, 1, stride=1, padding=0, bias=False)
        self.conv4 = nn.ConvTranspose2d(256,128, 4, stride=2, padding=1, bias=False)
        self.conv5 = nn.ConvTranspose2d(128,  3, 4, stride=2, padding=1)

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

def get_autoencoder(model, decoder, arch='ResNet18'):
    if arch in ['ResNet18']:
        return ResNetAutoEncoder(model, decoder)

class NMPicker:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter_counter = iter(dataloader)
        self.batch_data, self.batch_label = self._get_new_batch()
        self.batch_index = 0
        self.batch_size = dataloader.batch_size
    
    def _get_new_batch(self):
        """
        Get the next batch of data, start over from the begining if needed
        """
        try:
            data, label = next(self.iter_counter)
        except StopIteration:
            self.iter_counter = iter(self.dataloader)
            data, label = next(self.iter_counter)
        return data.cuda(), label.cuda()

    def get_non_match_images(self, labels):
        """
        Return a batch of images with different label in labels
        returning dimension: N*C*H*W
        """
        results = []
        result_labels = []
        for cur_label in labels:
            while cur_label == self.batch_label[self.batch_index]:  # See if the label matches
                self.batch_index += 1
                if self.batch_index >= len(self.batch_label):
                    self.batch_index = 0
                    self.batch_data, self.batch_label = self._get_new_batch()
            results.append(self.batch_data[self.batch_index])
            result_labels.append(self.batch_label[self.batch_index])
        return torch.stack(results, dim=0).cuda()

def save_reconstruction(inputs, matched_results, nonmatch_results, save_dir, epoch):
    plt.figure(figsize=(25,25))
    plt.subplot(1,3,1)
    plt.axis("off")
    plt.title("Original Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(inputs, padding=2, normalize=True).cpu(),(1,2,0)))

    plt.subplot(1,3,2)
    plt.axis("off")
    plt.title("Match Reconstruction")
    plt.imshow(np.transpose(matched_results,(1,2,0)))

    plt.subplot(1,3,3)
    plt.axis("off")
    plt.title("NonMatch Reconstruction")
    plt.imshow(np.transpose(nonmatch_results,(1,2,0)))

    plt.savefig(save_dir + os.sep + str(epoch) + ".png")


def save_hist(match_scores, nonmatch_scores, save_dir):
    max_score = max(match_scores + nonmatch_scores)
    min_score = min(match_scores + nonmatch_scores)
    bins = np.linspace(min_score, max_score, 100)
    plt.figure(figsize=(10,10))
    plt.hist(match_scores, bins, alpha=0.5, label='Matched')
    plt.hist(nonmatch_scores, bins, alpha=0.5, label='Non Matched')
    plt.legend(loc='upper right')
    plt.tight_layout()
    histo_file = os.path.join(save_dir, "histo.png")
    plt.savefig(histo_file)
    print(f"Fig save to {histo_file}")


def train_autoencoder(autoencoder, train_loader, optimizer_decoder, alpha, num_classes, device="cuda", start_epoch=0, max_epochs=50, verbose=True, save_output=False, arch='classifier32', train_in_eval_mode=False):
    assert start_epoch < max_epochs
    avg_loss = 0.
    avg_match_loss = 0.
    avg_nonmatch_loss = 0.

    # always use 1-aplha
    nonmatch_ratio = 1.-alpha

    criterion = nn.L1Loss()

    for epoch in range(start_epoch, max_epochs):
        for phase in ['train']:
            # if phase == "train" and epoch < float(max_epochs)/2 and not train_in_eval_mode:
            if phase == "train":
                # scheduler_decoder.step()
                autoencoder.train()
                autoencoder.encoder.eval()
            else:
                autoencoder.eval()

            running_match_loss = 0.0
            running_nonmatch_loss = 0.0
            count = 0

            if verbose:
                pbar = tqdm(train_loader, ncols=80)
            else:
                pbar = train_loader

            nmpicker = NMPicker(train_loader)

            for batch, data in enumerate(pbar):
                inputs, labels = data
                count += inputs.size(0)
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer_decoder.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    dist = torch.zeros((inputs.shape[0], num_classes-1)).to(inputs.device) + (1/float(num_classes-1))
                    perm = torch.multinomial(dist,1).reshape(-1)
                    nm_labels = torch.remainder(labels + perm + 1, num_classes)
                    nonmatch_outputs = autoencoder(inputs, nm_labels)
                    nonmatch_target = nmpicker.get_non_match_images(labels)
                    nonmatch_loss = criterion(nonmatch_outputs, nonmatch_target)

                    matched_outputs = autoencoder(inputs, labels)
                    match_loss = criterion(matched_outputs, inputs)
                    
                    if save_output:
                        save_dir = f"c2ae_results/reconstruction_alpha_{alpha}_epoch_{max_epochs}_phase_{phase}_arch_{arch}_trainineval_{train_in_eval_mode}"
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, mode=0o777)
                        
                        if epoch==max_epochs-1:
                            # Save the histogram 
                            if batch==0:
                                match_scores = []
                                nonmatch_scores = []
                            match_scores += (torch.abs(matched_outputs - inputs).view(inputs.shape[0],-1).mean(1).detach().cpu()).tolist()
                            nonmatch_scores += (torch.abs(nonmatch_outputs - inputs).view(inputs.shape[0],-1).mean(1).detach().cpu()).tolist()

                            if (batch == len(train_loader)-1):
                                save_hist(match_scores, nonmatch_scores, save_dir)

                        if (batch == len(train_loader)-1):
                            matched_results = torchvision.utils.make_grid(matched_outputs.detach().cpu(), padding=2, normalize=True)
                            nonmatch_results = torchvision.utils.make_grid(nonmatch_outputs.detach().cpu(), padding=2, normalize=True)

                            save_reconstruction(inputs, matched_results, nonmatch_results, save_dir, epoch)                            

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

    # Save min error for training examples
    if save_output:
        autoencoder.eval()
        scores_lst = []

        with torch.set_grad_enabled(False):
            for batch, data in enumerate(pbar):
                last_batch = (batch == len(train_loader)-1)

                inputs, labels = data
                k_labels = torch.arange(num_classes).unsqueeze(1).expand(-1, inputs.shape[0])
                
                count += inputs.size(0)
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    # matched_outputs = autoencoder(inputs, labels)
                    # match_loss = criterion(matched_outputs, inputs) 
                    for class_i in range(num_classes):
                        reconstruction_i = autoencoder(inputs, k_labels[class_i]).detach().cpu()
                        errors = torch.abs(inputs.cpu() - reconstruction_i).view(inputs.shape[0], -1).mean(1)
                        if class_i == 0:
                            min_errors = errors
                        else:
                            min_errors = torch.min(min_errors, errors)

                        if last_batch:
                            if class_i == 0:
                                reconstructions = [reconstruction_i.cpu()]
                            else:
                                reconstructions += [reconstruction_i.cpu()]
                    scores_lst += min_errors.tolist()
                        
                    if verbose:
                        pbar.set_postfix(count=count)

            plt.figure(figsize=(25,25))
            plt.subplot(5,3,1)
            plt.axis("off")
            plt.title("Original Images")
            plt.imshow(np.transpose(torchvision.utils.make_grid(inputs, padding=2, normalize=True).cpu(),(1,2,0)))
            
            for class_i, reconstruction_i in enumerate(reconstructions):
                results = torchvision.utils.make_grid(reconstruction_i.detach(), padding=2, normalize=True)
                plt.subplot(5,3,2+class_i)
                plt.axis("off")
                plt.title(f"Reconstruction with class {class_i}")
                plt.imshow(np.transpose(torchvision.utils.make_grid(results, padding=2, normalize=True).cpu(),(1,2,0)))
            
            plt.savefig(save_dir + os.sep + "reconstructions.png")

            max_score = max(scores_lst)
            min_score = min(scores_lst)
            bins = np.linspace(min_score, max_score, 100)
            plt.figure(figsize=(10,10))
            plt.hist(scores_lst, bins, alpha=0.5, label='Min_error')
            plt.legend(loc='upper right')
            # plt.show()
            plt.tight_layout()
            histo_file = os.path.join(save_dir, "minerror_histo.png")
            plt.savefig(histo_file)
            print(f"Fig save to {histo_file}")

    return avg_match_loss, avg_nonmatch_loss


