# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:27:18 2020

@author: hiltunh3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Poisson, kl_divergence as kl

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable
from typing import Optional, List, Tuple, Union, Iterable

from gaussian_loss import GaussianLoss, gaussian_log_likelihood, kl_div
from multiEncoder import MultiInputLayer

class Encoder(nn.Module):
    def __init__(self,
                 n_head: int,
                 input_dim_list: List[int],
                 hidden_dim: int=128, 
                 z_dim=2):
        '''
        Args:
            input_dim: A integer indicating the size of input
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()

        self.encoders = nn.ModuleList(
            [ 
                nn.Sequential(
                    nn.Linear(input_dim_list[i], hidden_dim),
                    nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001),
                    nn.ReLU(),
                )
                for i in range(n_head)
            ]
        )
        
        self.shared_layer = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001),
                    nn.ReLU(),
                )
        
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, head_id: int):
        # x is of shape [batch_size, input_dim]
        hidden = self.encoders[head_id](x)
        hidden = self.shared_layer(hidden)
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim_list: List[int], n_head: int):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension 
        '''
        super().__init__()
        
        #self.shared_layer = nn.Sequential(
        #            nn.Linear(z_dim, hidden_dim),
        #            nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001),
        #            nn.ReLU(),
        #        )
        
        #self.decoders = nn.ModuleList(
        #    [ 
        self.decoders = nn.Sequential(
                    nn.Linear(z_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim_list[0]),
                )
        #        for i in range(n_head)
        #    ]
        #)
    

    def forward(self, x, head_id: int):
        # x is of shape [batch_size, latent_dim]
        #predicted = self.shared_layer(x)
        # predicted is of shape [batch_size, output_dim]
        predicted = self.decoders(x)
        return predicted

class Classifier(nn.Module): # change for multiple
    def __init__(self, n_input, hidden_dim, n_labels):
        
        super().__init__()
        self.n_input = n_input
        self.hidden_dim = hidden_dim
        self.n_classes = n_labels
        
        self.classifier = nn.Sequential(
            nn.Linear(n_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_labels),          
        )
    
    def loss(self, latent_tensors, predict_true):
        losses = []
        for i, z in enumerate(latent_tensors): 
            cls_logits = nn.LogSoftmax(dim=1)(self.classifier(z)) # why log
            if predict_true:
                cls_target = torch.zeros(
                    self.n_classes, dtype=torch.float32, device=z.device
                )
                cls_target[i] = 1.0
            else:
                cls_target = torch.ones(
                    self.n_classes, dtype=torch.float32, device=z.device
                ) / (self.n_classes - 1)
                cls_target[i] = 0.0

            l_soft = cls_logits * cls_target
            cls_loss = -l_soft.sum(dim=1).mean()
            losses.append(cls_loss)

        total_loss = torch.stack(losses).sum()
        return total_loss
    
    def classify(self, z):
        p = self.classifier(z)
        p = nn.Softmax(dim=-1)(p)
        return p
        
    def forward(self, x):
        return self.classifier(x)

class VAE(nn.Module): # needs n_labels as a parameter
    def __init__(self, enc, dec, z_dim=2, n_labels=8, n_shared=3, n_other=4):
        ''' This the VAE, which takes a encoder and decoder.
        '''
        super().__init__()

        self.enc = enc
        self.dec = dec
        self.n_shared = n_shared
        self.n_other = n_other
        
            
    def reparam(self, z_mu, z_var):
        """Reparameterisation trick to sample z values. 
        This is stochastic during training,  and returns the mode during evaluation."""
        
        if self.training:
            std = torch.exp(z_var.mul(0.5))
            eps = torch.randn_like(std)
            return eps.mul(std).add_(z_mu)
        else:
            return z_mu
        
    def loss(self, reconst_x, x, z_mu, z_var, n_head: int):
        loss = GaussianLoss(x.size(1))
        #recon_loss = loss(x, reconst_x)
        #recon_loss = torch.sum(recon_loss)
        
        cols_data = [j for j in range(0, self.n_shared + self.n_other)]
        cols_output = [j for j in range(0, self.n_shared)] + \
                    [(self.n_shared + n_head*self.n_other + j) for j in range(0, self.n_other)]
        #print(cols_output)   #[0, 1, 2, 3, 4, 5, 6] or  [0, 1, 2, 7, 8, 9, 10]
        x_data = x[:, cols_data]
        x_mean = reconst_x[:, cols_output]
        
        recon_loss = loss(x_data, x_mean)
        recon_loss = torch.sum(recon_loss)
        
        #recon_loss = F.mse_loss(x_mean, x_data, reduction='sum')
        # kl divergence loss
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu.pow(2) - 1.0 - z_var)
        #kl_loss = kl_div(z_mu, torch.exp(z_var))
        #kl_loss = torch.sum(kl_loss)
        
        #kl_mixture = log_normal_mixture(z_given_x, prior[0], prior[1])

        # total loss
        return (recon_loss + kl_loss), recon_loss, kl_loss
    
    def get_z(self, x, head_id: int):
        """Encode a batch of data points, x, into their z representations."""
        
        z_mu, z_var = self.enc(x, head_id)
        return self.reparam(z_mu, z_var)


    def forward(self, x, head_id: int):
        
        # encode
        z_mu, z_var = self.enc(x, head_id)
        

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        x_sample = self.reparam(z_mu, z_var)
              
        # decode
        predicted = self.dec(x_sample, head_id)
        return predicted, z_mu, z_var

# # multiencoder

class multiEncoder(nn.Module):
    def __init__(self,
                 n_head: int,
                 shared_dim: int, 
                 unique_dim: int,
                 hidden_dim: int=128, 
                 z_dim=2):
        '''
        Args:
            input_dim: A integer indicating the size of input
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()
        self.layer1 = MultiInputLayer(shared_dim, unique_dim, n_head, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu = nn.Linear(hidden_dim, z_dim)

        self.var = nn.Linear(hidden_dim, z_dim)


    def forward(self, x_shared, x_unique, head_id: int):
        # x is of shape [batch_size, input_dim]
        hidden = F.relu(self.layer1(x_shared, x_unique, head_id))
        hidden = F.relu(self.layer2(hidden))
        #hidden = self.shared_layer(hidden)
        #how this works, can just concat these?
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var

class multiVAE(nn.Module): 
    def __init__(self, enc, dec, z_dim=2, n_labels=8, n_shared=3, n_other=4):
        super().__init__()

        self.enc = enc
        self.dec = dec
        self.n_shared = n_shared
        self.n_other = n_other
        
            
    def reparam(self, z_mu, z_var):
        """Reparameterisation trick to sample z values. 
        This is stochastic during training,  and returns the mode during evaluation."""
        
        if self.training:
            std = torch.exp(z_var.mul(0.5))
            eps = torch.randn_like(std)
            return eps.mul(std).add_(z_mu)
        else:
            return z_mu
        
    def loss(self, reconst_x, x, z_mu, z_var, n_head: int):
        # reconstruction loss
        #recon_loss = F.mse_loss(reconst_x, x, reduction='sum')
        loss = GaussianLoss(x.size(1))
        
        cols_data = [j for j in range(0, self.n_shared + self.n_other)]
        cols_output = [j for j in range(0, self.n_shared)] + \
                    [(self.n_shared + n_head*self.n_other + j) for j in range(0, self.n_other)]
        #print(cols_output)   #[0, 1, 2, 3, 4, 5, 6] or  [0, 1, 2, 7, 8, 9, 10]
        x_data = x[:, cols_data]
        x_mean = reconst_x[:, cols_output]
        
        recon_loss = loss(x_data, x_mean) 
        recon_loss = torch.sum(recon_loss)
        #recon_loss = F.mse_loss(x_mean, x_data, reduction='sum')
        # kl divergence loss
        #kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu.pow(2) - 1.0 - z_var)
        kl_loss = kl_div(z_mu, torch.exp(z_var))
        kl_loss = torch.sum(kl_loss)
      
        # total loss
        return (recon_loss + kl_loss), recon_loss, kl_loss
    
    def get_z(self, x, head_id: int):
        """Encode a batch of data points, x, into their z representations."""
        cols_shared = [j for j in range(0, self.n_shared)]
        cols_unique = [j for j in range(self.n_shared, self.n_shared+self.n_other)]

        x_shared = x[:, cols_shared]
        x_unique = x[:, cols_unique]
        
        z_mu, z_var = self.enc(x_shared, x_unique, head_id)
        return self.reparam(z_mu, z_var)


    def forward(self, x, head_id: int):
        cols_shared = [j for j in range(0, self.n_shared)]
        cols_unique = [j for j in range(self.n_shared, self.n_shared+self.n_other)]
        x_shared = x[:, cols_shared]
        x_unique = x[:, cols_unique]
        
        # encode
        z_mu, z_var = self.enc(x_shared, x_unique, head_id)
        

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        z_sample = self.reparam(z_mu, z_var)
              
        # decode
        predicted = self.dec(z_sample, head_id)
        return predicted, z_mu, z_var
