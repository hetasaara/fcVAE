# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:56:09 2020

@author: hiltunh3
"""

import numpy as np
import torch
from fcvae_dataset import load_data
from fcvae_model import VAE, Encoder, Decoder, Classifier
from fcvae_trainer import fcTrainer

import itertools
import pandas

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Torch version:", torch.__version__)
print("Device:", device)

def validation_accuracy(model, discriminator, test_iterators):
    model.eval()
    discriminator.eval()
    probs = []
    for tensors in test_iterators:
        ps = []
        for head_id, batch in enumerate(tensors):
            z = model.get_z(batch.float(), head_id)
            p = discriminator.classify(z)
            p = p.cpu().detach().numpy()
            ps.append(p)
        #pr = torch.cat(ps).cpu().detach().numpy()
        #print(pr.shape)
        probs.append(ps)
    return probs

def validation_loss(test_iterators, model, discriminator, input_dim_list, weight):
    # set the evaluation mode
    model.eval()
    discriminator.eval()
    n_head = len(input_dim_list)
    # test loss for the data
    test_losses = []
    d_losses = []
    for tensors in test_iterators:
        latent_samples = []
        for head_id, batch in enumerate(tensors):
                z = model.get_z(batch.float(), head_id)
                latent_samples.append(z)
        d_loss = discriminator.loss([t.detach() for t in latent_samples], True)
        d_loss *= weight
        d_losses.append(d_loss)
        for head_id, x in enumerate(tensors): 
            # reshape the data
            x = x.view(-1, input_dim_list[head_id]) 
            x = x.float()
            x = x.to(device)

            # forward pass
            x_sample, z_mu, z_var = model(x, head_id)

            loss, recs, kls = model.loss(x_sample, x, z_mu, z_var, head_id)
            test_losses.append(loss)
    sum_loss = torch.stack(test_losses).sum()
    disc = torch.stack(d_losses).sum()
    
    return sum_loss, disc

# optimal hyperparameters
filepath1 = "/scratch/cs/csb/projects/single-cell-analysis/FCM/AML_FCM/data_files1/validation_sets/"

ks = np.arange(0,150,20)
params = [ks, [2,4,6], [32,64,128]]
params = list(itertools.product(*params))
print(params)
accuracies = []
val_losses = []
disc_losses = []
train_probs = []
for p in params:
    batch_size = 256         # number of data points in each batch
    n_epochs = 30           # times to run the model on complete data
    hidden_dim = 128        # hidden dimension
    d_hidden_dim = p[2]
    latent_dim = p[1]        # latent vector dimension
    lr = 1e-3              # learning rate
    n_head = 6
    n_shared = 3
    missing = 20

    datasets = load_data(filepath1)

    input_dim_list = []
    output_dim_list = []
    train_iterators = []
    test_iterators = []
    train_cells = 0
    test_cells = 0
    for tube in datasets:
        input_dim_list.append(len(tube[0].columns))
        output_dim_list.append(len(tube[0].columns)+missing)
        train_iterators.append(tube[3])
        test_iterators.append(tube[4])
        train_cells += tube[1].nb_cells
        test_cells += tube[2].nb_cells
    
    N_train = int(train_cells/n_head)
    N_test = int(test_cells/n_head)
    print(p)
        
    encoder = Encoder(n_head, input_dim_list, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, d_hidden_dim, output_dim_list, n_head)
    model = VAE(encoder, decoder).to(device)
    discriminator = Classifier(latent_dim, hidden_dim, n_head) 

    trainer = fcTrainer(
            model,
            discriminator,
            train_loaders = train_iterators,
            test_loaders = test_iterators,
            N_train = N_train,
            N_test = N_test,
            dloss_weight = p[0],
            device = device,
            n_epochs = n_epochs,
            print_frequency = 1,
            optimizer_kwargs = dict(lr = lr)
    )

    trainer.train()
    prob1 = [item[0] for item in trainer.TRAIN_PROBS]
    train_probs.append(prob1)
    acc = validation_accuracy(model, discriminator, zip(*test_iterators))
    acc = [item for sublist in acc for item in sublist]
    acc = [(acc[i::n_head]) for i in range(n_head)]
    correct = 0
    for i, a in enumerate(acc):
        conc = [item for sublist in a for item in sublist]
        conc = np.array(conc)
        for r in range(len(conc)):
            if conc[r,i] == max(conc[r,:]):
                correct += 1

    accuracy = correct/(N_test*n_head)
    accuracies.append(accuracy)
    val_loss, disc_loss = validation_loss(zip(*test_iterators), model, discriminator, input_dim_list, p[0])
    val_losses.append(val_loss/N_test)
    disc_losses.append(disc_loss/N_test)
print(accuracies)
print(val_losses)
print(disc_losses)

val_losses = torch.stack(val_losses).cpu().detach().numpy()
disc_losses = torch.stack(disc_losses).cpu().detach().numpy()

# +
train_probs = pandas.DataFrame(train_probs)
train_probs.to_csv("/scratch/cs/csb/projects/single-cell-analysis/FCM/AML_FCM/training_probs_grid_.csv", sep=',',index=False)

df = pandas.DataFrame(data={"rec": val_losses, "acc": accuracies, "disc": disc_losses})
df.to_csv("/scratch/cs/csb/projects/single-cell-analysis/FCM/AML_FCM/validation_metrics_true.csv", sep=',',index=False)
# -


