# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:36:11 2020

@author: hiltunh3
"""

import numpy as np
import torch
from fcvae_model import VAE, Classifier
from fcvae_plot import plot_trainer_history, plot_probabilities, CellPlotter2D
from fcvae_logger import Logger
from umap import UMAP
from sklearn.decomposition import PCA


class fcTrainer:
    """
    The main trainer class.
    """

    def __init__(
            self,
            vae: VAE,
            discriminator: Classifier,
            train_loaders,
            test_loaders,
            N_train,
            N_test,
            device,
            optimizer_kwargs,
            n_epochs: int = 100,
            print_frequency: int = 1,
            dloss_weight = 50.0
    ):

        # Logging
        self.epoch = 0
        self.n_epochs = n_epochs
        self.print_frequency = print_frequency
        self.logger = Logger(verbose=True, zfill=len(str(self.n_epochs)))
        self.device = device
        self.logger.message('Creating a trainer on device `' + str(self.device) + "`")

        # Model and data
        self.vae = vae.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.train_iterators = train_loaders
        self.test_iterators = test_loaders
        self.n_tubes = discriminator.n_classes
        self.N_train = N_train
        self.N_test = N_test
        self.dloss_weight = dloss_weight

        # Create optimizers
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), **optimizer_kwargs)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), **optimizer_kwargs)

        # Create store for loss values
        self.TRAIN_HISTORY = np.zeros((self.n_epochs, 5))
        self.TEST_HISTORY = np.zeros((self.n_epochs, 5))
        
        self.TRAIN_PROBS = []
        self.TEST_PROBS = []

    def train(self):
        """
        Train the model
        """
        self.vae.train()
        self.discriminator.train()
        self.logger.message("TRAINING STARTED (n_epochs = " + str(self.n_epochs) + ").")

        # Loop over epochs
        for i in range(self.n_epochs):
            self.epoch = i
            self.epoch_train()
            self.epoch_test()
            self.epoch_end()

        self.logger.message("TRAINING FINISHED.\n")
        self.vae.eval()
        self.discriminator.eval()

    def epoch_train(self):
        """
        Performs one sweep through the training set and computes average training loss
        and different terms and stores them in self.TRAIN_HISTORY
        """
        self.vae.train()
        self.discriminator.train()
        
        #n_head = len(input_dim_list)  needed?
        total_loss = []
        d_losses = []
        fool_losses = []
        mean_ps = []
        recs = 0
        kls = 0
        
        for i, tensors in enumerate(zip(*self.train_iterators)):
             # Train discriminator
            train_discriminator = True 
            if train_discriminator:
                latent_samples = []
                for head_id, x in enumerate(tensors):
                    z_sample = self.vae.get_z(x.float().to(self.device), head_id)
                    latent_samples.append(z_sample)
                    prediction = self.discriminator.classify(z_sample)
                    pred = prediction.detach().cpu().numpy()
                    pred = [item for sublist in pred for item in sublist]
                    if head_id == 0:
                        mean_ps.append(pred)
               
                d_loss = self.discriminator.loss([t.detach() for t in latent_samples], True) # minimize dataset mixing, detach?
                d_loss *= self.dloss_weight
                d_losses.append(d_loss)
                self.disc_optimizer.zero_grad()
                d_loss.backward()
                self.disc_optimizer.step()
    
                # Train vae to fool discriminator
                fool_loss = self.discriminator.loss(latent_samples, False) # minimize disc accuracy
                fool_loss*= self.dloss_weight
                fool_losses.append(fool_loss)
                self.vae_optimizer.zero_grad()
                fool_loss.backward()
                self.vae_optimizer.step()
            
            # Train vae
            train_losses = []
            for head_id, x in enumerate(tensors):
                # reshape the data
                #x = x.view(-1, input_dim_list[head_id]) needed?
                x = x.float()
                x = x.to(self.device) 

                # forward pass
                x_sample, z_mu, z_var = self.vae(x, head_id)

                loss, rec, kl = self.vae.loss(x_sample, x, z_mu, z_var, head_id)

                train_losses.append(loss)
                total_loss.append(loss)
                recs += rec
                kls += kl.item()

            sum_loss = torch.stack(train_losses).sum()
            # update the gradients to zero
            self.vae_optimizer.zero_grad() 
            # backward pass
            sum_loss.backward()
            # update the weights
            self.vae_optimizer.step()
        total_loss = torch.stack(total_loss).sum()
        fool = torch.stack(fool_losses).sum()
        disc = torch.stack(d_losses).sum()

        # Average loss terms over whole training set
        losses = np.array([total_loss, kls, recs, fool, disc])/self.N_train
        p = [item for sublist in mean_ps for item in sublist]
        p = [np.mean(p[::self.n_tubes]), np.mean(p[1::self.n_tubes])]
        self.TRAIN_HISTORY[self.epoch, :] = losses
        self.TRAIN_PROBS.append(p)

    @torch.no_grad()
    def epoch_test(self, direction: int = 0):
        """
        Computes average loss on the validation set and stores it in self.VALID_HISTORY
        """
        self.vae.eval()
        self.discriminator.eval()
        
        total_loss = []
        d_losses = []
        fool_losses = []
        mean_ps = []
        recs = 0
        kls = 0
        
        for i, tensors in enumerate(zip(*self.test_iterators)):
             # Train discriminator
            train_discriminator = True 
            if train_discriminator:
                latent_samples = []
                for head_id, x in enumerate(tensors):
                    z_sample = self.vae.get_z(x.float().to(self.device), head_id)
                    latent_samples.append(z_sample)
                    prediction = self.discriminator.classify(z_sample)
                    pred = prediction.detach().cpu().numpy()
                    pred = [item for sublist in pred for item in sublist]
                    if head_id == 0:
                        mean_ps.append(pred)
               
                d_loss = self.discriminator.loss([t.detach() for t in latent_samples], True) # minimize dataset mixing, detach?
                d_loss *= self.dloss_weight
                d_losses.append(d_loss)
    
                # Train vae to fool discriminator
                fool_loss = self.discriminator.loss(latent_samples, False) # minimize disc accuracy
                fool_loss*= self.dloss_weight
                fool_losses.append(fool_loss)

            
            # Train vae
            train_losses = []
            for head_id, x in enumerate(tensors):
                # reshape the data
                #x = x.view(-1, input_dim_list[head_id]) needed?
                x = x.float()
                x = x.to(self.device) 

                # forward pass
                x_sample, z_mu, z_var = self.vae(x, head_id)

                loss, rec, kl = self.vae.loss(x_sample, x, z_mu, z_var, head_id)

                train_losses.append(loss)
                total_loss.append(loss)
                recs += rec
                kls += kl.item()
            sum_loss = torch.stack(train_losses).sum()
        total_loss = torch.stack(total_loss).sum()
        fool = torch.stack(fool_losses).sum()
        disc = torch.stack(d_losses).sum()

        # Average loss terms over whole test set
        losses = np.array([total_loss, kls, recs, fool, disc])/self.N_test
        p = [item for sublist in mean_ps for item in sublist]
        p = [np.mean(p[::self.n_tubes]), np.mean(p[1::self.n_tubes])]
        self.TEST_PROBS.append(p)
        self.TEST_HISTORY[self.epoch, :] = losses

    @torch.no_grad()
    def epoch_end(self):
        """
        Things to be done at the end of each epoch
        """
        i = self.epoch
        train_loss_vae = self.TRAIN_HISTORY[i, 0]
        test_loss_vae = self.TEST_HISTORY[i, 0]
        train_loss_disc = self.TRAIN_HISTORY[i, 4]
        test_loss_disc = self.TEST_HISTORY[i, 4]

        # Print average training and validation loss over epoch
        if (i % self.print_frequency == 0) or (i == self.n_epochs - 1):
            self.logger.train_info(i, train_loss_vae, test_loss_vae, train_loss_disc, test_loss_disc)

    @torch.no_grad()
    def plot(self, i_start: int = 0, save_name=None):
        """
        Plot training history
        """
        if self.n_epochs >= 2:
            plot_trainer_history(self, i_start, save_name)
        else:
            self.logger.message('Skipped plotting history because n_epochs <= 1.')
            
    @torch.no_grad()
    def plot_probs(self, i_start: int = 0, save_name=None):
        """
        Plot training history
        """
        if self.n_epochs >= 2:
            plot_probabilities(self, i_start, save_name)
        else:
            self.logger.message('Skipped plotting history because n_epochs <= 1.')

    @torch.no_grad()
    def create_plotter(self, color_by_type: bool = True, embedding=None):
        """
        Create a 2D plotter for visualization
        """
        z = self.get_latent_numpy()
        d = z.shape[1]
        N = len(self.dataset)
        if color_by_type:
            colors = self.dataset.cell_colors
            labels = self.dataset.cell_types
        else:
            colors = [("#1f77b4" if i in self.train_indices else "#ff7f0e") for i in range(0, N)]
            colors = np.array(colors)
            labels = [("Train set" if i in self.train_indices else "Test set") for i in range(0, N)]

        # Embed to 2d
        if embedding is not None:
            if d == 2:
                raise ValueError("embedding must be None, when latent dimension is 2!")

            if embedding == 'UMAP':
                self.logger.message("Computing UMAP embedding")
                reducer = UMAP()
                z = reducer.fit_transform(z)
            elif embedding == "PCA":
                self.logger.message("Computing PCA embedding")
                reducer = PCA(n_components=2)
                z = reducer.fit_transform(z)
            else:
                raise ValueError("embedding must be PCA or UMAP")
        else:
            if d != 2:
                raise ValueError("embedding cannot be None, when latent dimension > 2")

        # Create plotter object
        plotter = CellPlotter2D(z, colors, labels, embedding)
        return plotter
