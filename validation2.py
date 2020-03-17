# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:17:43 2020

@author: hiltunh3
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:56:09 2020

@author: hiltunh3
"""

import numpy as np
import torch
from fcvae_dataset import load_data
from fcvae_model import multiVAE, multiEncoder, Decoder, Classifier
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
            ps.append(p)
        pr = torch.cat(ps).cpu().detach().numpy()
        probs.append(pr)
    return probs

def validation_loss(test_iterators, model, input_dim_list):
    # set the evaluation mode
    model.eval()
    n_head = len(input_dim_list)
    # test loss for the data
    test_losses = []
    
    for tensors in test_iterators:
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

    return sum_loss

# optimal hyperparameters
filepath1 = "/scratch/cs/csb/projects/single-cell-analysis/FCM/AML_FCM/data_files1/validation_sets/"

ks = np.arange(0,150,20)
params = [ks, [2,4,6], [32,64,128]]
params = list(itertools.product(*params))
print(params)
accuracies = []
val_losses = []
for i in params:
    batch_size = 256         # number of data points in each batch
    n_epochs = 30           # times to run the model on complete data
    hidden_dim = 128        # hidden dimension
    d_hidden_dim = i[2]
    latent_dim = i[1]        # latent vector dimension
    lr = 1e-3              # learning rate
    n_head = 6
    n_shared = 3
    n_unique = 4
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

    input_dim_list.append(n_shared)        
    
    N_train = int(train_cells/n_head)
    N_test = int(test_cells/n_head)
    print(i)
        
    encoder = multiEncoder(n_head, n_shared, n_unique, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, d_hidden_dim, output_dim_list, n_head)
    model = multiVAE(encoder, decoder).to(device)
    discriminator = Classifier(latent_dim, hidden_dim, n_head) 

    trainer = fcTrainer(
            model,
            discriminator,
            train_loaders = train_iterators,
            test_loaders = test_iterators,
            N_train = N_train,
            N_test = N_test,
            dloss_weight = i[0],
            device = device,
            n_epochs = n_epochs,
            print_frequency = 1,
            optimizer_kwargs = dict(lr = lr)
    )

    trainer.train()
    acc = validation_accuracy(model, discriminator, zip(*test_iterators))
    acc = [item for sublist in acc for item in sublist]
    acc = [item for sublist in acc for item in sublist]
    correct = 0
    probs1 = acc[::n_head]
    for i in probs1[:N_test]:
        if i > 1/n_head:
            correct += 1
    for i in probs1[N_test:]:
        if i < 1/n_head:
            correct += 1

    accuracy = correct/len(acc[::n_head])
    accuracies.append(accuracy)
    val_loss = validation_loss(zip(*test_iterators), model, input_dim_list)/N_test
    val_losses.append(val_loss)
print(accuracies)
print(val_losses)

val_losses = torch.stack(val_losses).cpu().detach().numpy()

df = pandas.DataFrame(data={"rec": val_losses, "acc": accuracies})
df.to_csv("/scratch/cs/csb/projects/single-cell-analysis/FCM/AML_FCM/validation_metrics2.csv", sep=',',index=False)


