# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:51:58 2018

@author: David
"""

import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torchvision

from d4utils import CODES_DE_SECTION
from d4utils import VolcanoesConv
from d4utils import VolcanoesDataset, VolcanoesLoader
from d4utils import compute_accuracy, compute_confusion_matrix


# TODO Logistique
# Mettre 'BACC' ou 'GRAD'
SECTION = 'GRAD'

searches_train = np.load("train_searches.npy")
searches_val = np.load("valid_searches.npy")
document_labels = np.load("document_labels.npy")
search_features = np.load("searches_features.npy")
a = np.load("train_correspondance.npy")
a[0, 1]
# TODO Logistique
# Mettre son numéro d'équipe ici
NUMERO_EQUIPE = 45

# Crée la random seed
RANDOM_SEED = CODES_DE_SECTION[SECTION] + NUMERO_EQUIPE


class SearchEngine(nn.Module):
    """
    This class defines a neural network capable of predicting documents directly
    """

    def __init__(self, in_features = 3000, num_documents = 5689, hidden_layer_size = 200):
        super().__init__()
        
        # Defines two linear hidden layers
        self.L1 = nn.Linear(in_features = 3000, out_features = hidden_layer_size)
        self.L2 = nn.Linear(in_features = hidden_layer_size, out_features = hidden_layer_size)
        self.L3 = nn.Linear(in_features = hidden_layer_size, out_features = hidden_layer_size)
        self.L4 = nn.Linear(in_features = hidden_layer_size, out_features = hidden_layer_size)
        self.L5 = nn.Linear(in_features = hidden_layer_size, out_features = hidden_layer_size)
        
        # Normalization layer (identical for all hidden layers)
        self.N = nn.BatchNorm1d(num_features = hidden_layer_size)
        
        # Output layer - out output for each unique document in the training set
        self.L6 = nn.Linear(in_features = hidden_layer_size, out_features = num_documents)

    def forward(self, x):

        y = self.L1(x)
        
        y = F.relu(self.N(y))
        y = self.L2(y)
        y = F.relu(self.N(y))
        y = self.L3(y)
        y = F.relu(self.N(y))
        y = self.L4(y)
        y = F.relu(self.N(y))
        y = self.L5(y)
        y = F.relu(self.N(y))
        
        y = self.L6(y)
        y = F.sigmoid(y)
        
        return y

if __name__ == '__main__':
    # Définit la seed
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # Des effets stochastiques peuvent survenir
    # avec cudnn, même si la seed est activée
    # voir le thread: https://bit.ly/2QDNxRE
    torch.backends.cudnn.deterministic = True

    # Définit si cuda est utilisé ou non
    # mettre cuda pour utiliser un GPU
    device = 'cpu'

    # Définit les paramètres d'entraînement
    # Nous vous conseillons ces paramètres. 
    # Cependant, vous pouvez les changer
    nb_epoch = 10
    learning_rate = 0.01
    momentum = 0.9
    batch_size = 32

    # Charge les données d'entraînement et de test
    train_set = VolcanoesDataset('data/Volcanoes_train.pt.gz')
    test_set = VolcanoesDataset('data/Volcanoes_test.pt.gz')
    
    # Crée le dataloader d'entraînement
    train_loader = VolcanoesLoader(train_set, batch_size=batch_size, \
        balanced=True, random_seed=RANDOM_SEED)
    test_loader = VolcanoesLoader(test_set, batch_size=batch_size, 
        balanced=True, random_seed=RANDOM_SEED)

    # TODO Q1C 
    # Instancier un réseau VolcanoesNet
    # dans une variable nommée "model"
    model = VolcanoesNet()
    
    # Tranfert le réseau au bon endroit
    model.to(device)
    
    # TODO Q1C
    # Instancier une fonction d'erreur BinaryCrossEntropy
    # et la mettre dans une variable nommée criterion
    criterion = nn.BCELoss()

    # TODO Q1C 
    # Instancier l'algorithme d'optimisation SGD
    # Ne pas oublier de lui donner les hyperparamètres
    # d'entraînement : learning rate et momentum!
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum)

    # TODO Q1C
    # Mettre le réseau en mode entraînement
    model.train()

    # TODO Q1C
    # Remplir les TODO dans la boucle d'entraînement
    for i_epoch in range(nb_epoch):

        start_time, train_losses = time.time(), []
        for i_batch, batch in enumerate(train_loader):
            images, targets = batch

            images = images.to(device)
            targets = targets.to(device)
            # TODO Q1C 
            # Mettre les gradients à zéro
            optimizer.zero_grad()

            # TODO Q1C
            # Calculer:
            # 1. l'inférence dans une variable "predictions"
            # 2. l'erreur dans une variable "loss"
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            # TODO Q1C
            # Rétropropager l'erreur et effectuer
            # une étape d'optimisation
            loss.backward()
            optimizer.step()

            # Ajoute le loss de la batch
            train_losses.append(loss.item())

        print(' [-] epoch {:4}/{:}, train loss {:.6f} in {:.2f}s'.format(
            i_epoch+1, nb_epoch, np.mean(train_losses), time.time()-start_time))
        
        # sauvegarde le réseau
        torch.save(model.state_dict(), 'volcanoes_model.pt')

    # affiche le score à l'écran
    #train_acc = compute_accuracy(model, train_loader, device)
    #print(' [-] train acc. {:.6f}%'.format(train_acc * 100))
    
    test_acc = compute_accuracy(model, test_loader, device)
    print(' [-] test acc. {:.6f}%'.format(test_acc * 100)) #85.995370%
    
    # affiche la matrice de confusion à l'écran
    matrix = compute_confusion_matrix(model, test_loader, device)
    print(' [-] conf. mtx. : \n{}'.format(matrix))
