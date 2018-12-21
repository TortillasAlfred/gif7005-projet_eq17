# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 11:00:06 2018

@author: David
"""
import time
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import pickle

def model_stats(model, X_train, y_train, X_valid, y_valid):
    score_train = model.score(X_train, y_train)
    score_valid = model.score(X_valid, y_valid)
    score_valid_200 = model.score(X_valid, y_valid, n_outputs = 200)
    total_epochs = model.total_epochs
    mean_epoch_time = np.mean(model.epoch_times)
    
    print("Training score: " + str(round(score_train, 5)))
    print("Validation score: " + str(round(score_valid, 5)))
    print("Top 200 validation score: " + str(round(score_valid_200, 5)))
    print("Total epochs: " + str(total_epochs))
    print("Mean epoch time: " + str(round(mean_epoch_time, 5)))
    

class NeuralNetwork(nn.Module):
    """
    This class defines a neural network capable of predicting documents directly from query expressions
    """

    def __init__(self, in_features, out_features, hidden_layer_sizes = [100, 100], 
                 activation_functions = ["relu", "relu"], drop_rates = [0, 0, 0]):
        super().__init__()
        
        self.hidden = nn.ModuleList()

        if len(hidden_layer_sizes) == 1:
            #Hidden layer
            if drop_rates[0] > 0:
                self.hidden.append(nn.Dropout(p = drop_rates[0]))
            self.hidden.append(nn.Linear(in_features = in_features, out_features = hidden_layer_sizes[0]))
            if activation_functions[0] == "relu":
                self.hidden.append(nn.ReLU())
            elif activation_functions[0] == "sig":
                self.hidden.append(nn.Sigmoid())
            elif activation_functions[0] == "tanh":
                self.encoder.append(nn.Tanh())
            else:
                print("Activation function not recognized - Hidden layer 1")
            # Output layer - out output for each unique document in the training set
            self.output = nn.Linear(in_features = hidden_layer_sizes[0], out_features = out_features)
            
        else:
            # First hidden layer and normalization layer
            if drop_rates[0] > 0:
                self.hidden.append(nn.Dropout(p = drop_rates[0]))
            self.hidden.append(nn.Linear(in_features = in_features, out_features = hidden_layer_sizes[0]))
            if activation_functions[0] == "relu":
                self.hidden.append(nn.ReLU())
            elif activation_functions[0] == "sig":
                self.hidden.append(nn.Sigmoid())
            elif activation_functions[0] == "tanh":
                self.encoder.append(nn.Tanh())
            else:
                print("Activation function not recognized - Hidden layer 1")
            self.hidden.append(nn.BatchNorm1d(num_features = hidden_layer_sizes[0]))
            
            # Middle hidden layers
            for i in range(len(hidden_layer_sizes) - 1):
                if drop_rates[i + 1] > 0:
                    self.hidden.append(nn.Dropout(p = drop_rates[i + 1]))
                self.hidden.append(nn.Linear(in_features = hidden_layer_sizes[i], out_features = hidden_layer_sizes[i + 1]))
                if activation_functions[i + 1] == "relu":
                    self.hidden.append(nn.ReLU())
                elif activation_functions[i + 1] == "sig":
                    self.hidden.append(nn.Sigmoid())
                elif activation_functions[i + 1] == "tanh":
                    self.hidden.append(nn.Tanh())
                else:
                    print("Activation function not recognized - Hidden layer " + str(i + 2))
                self.hidden.append(nn.BatchNorm1d(num_features = hidden_layer_sizes[i + 1]))
               
            # Output layer - out output for each unique document in the training set
            if drop_rates[len(drop_rates) - 1] > 0:
                self.hidden.append(nn.Dropout(p = drop_rates[len(drop_rates) - 1]))
            self.output = nn.Linear(in_features = hidden_layer_sizes[len(hidden_layer_sizes) - 1], out_features = out_features)

    def forward(self, x):
        
        for layer in self.hidden:
            x = layer(x)
            
        y = self.output(x)
        return y

class SearchEngine():
    
    def __init__(self, in_features, out_features, hidden_layer_sizes = [100, 100], activation_functions = ["relu", "relu"], drop_rates = [0, 0, 0]):
        self.model = NeuralNetwork(in_features, out_features, hidden_layer_sizes, activation_functions, drop_rates) # Initializes neural network
        self.losses_train = list()
        self.unique_train = list() # Number of unique documents in the prediction set for the training data
        self.scores_train = list()
        self.scores_valid = list()
        self.total_epochs = 0
        self.epochs_plot = list()
        self.epoch_times = list()
        self.ewma_train = list()
        self.ewma_valid = list()
        
    def fit(self, X_train, y_train, X_train_orig, y_train_orig, X_valid_orig = list(), y_valid_orig = list(), score_interval = 5,
            device = "cpu", n_epochs = 10, optimizer = "sgd", learning_rate = 0.1, momentum = 0.9, weight = list(), early_stopping = True, ewma_coeff = 0.1, 
            min_slope = 0.001, min_score_valid = 0.13, saving = True):
        self.device = device
        self.model.to(self.device)
        self.score_interval = score_interval
        
        # Checks if validation data were provided
        if len(X_valid_orig) > 0 and len(y_valid_orig) > 0:
            self.validation = True
        else:
            self.validation = False
        # Puts model in training mode
        self.model.train()
        # Defines optimizer and error criterion
        if optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr = learning_rate, momentum = momentum)
        elif optimizer == "adadelta":
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr = learning_rate, rho = momentum)
        else:
            print("optimizer not recognized")
        if len(weight) == 0:
            weight = None
        else:
            weight = torch.FloatTensor(weight)
        criterion = nn.CrossEntropyLoss(weight = weight)
        
        # Used for the printing output - total number of outputs resulting after all iterations complete
        final_epochs = self.total_epochs + n_epochs
        
        # Gets searches and clicks, mounts to device
        train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        searches, clicks = train_data.tensors
        searches = searches.to(device)
        clicks = clicks.to(device)
        
        for i_epoch in range(n_epochs):
            self.total_epochs += 1
            start_time = time.time()
            optimizer.zero_grad()
            predictions = self.model(searches)
            loss = criterion(predictions, clicks)
            self.losses_train.append(loss.item())
            # Calculates the scores once per score interval
            if (i_epoch + 1) % score_interval == 0:
                self.scores_train.append(self.score(X_train_orig, y_train_orig))
                self.epochs_plot.append(self.total_epochs)
                if len(self.ewma_train) == 0:
                    self.ewma_train.append(self.scores_train[0])
                else:
                    self.ewma_train.append(ewma_coeff*self.scores_train[-1] + (1 - ewma_coeff)*self.ewma_train[-1])
                
                # Calculates validation scores if validation data were provided
                if self.validation:
                    self.scores_valid.append(self.score(X_valid_orig, y_valid_orig))
                    if len(self.ewma_valid) == 0:
                        self.ewma_valid.append(self.scores_valid[0])
                    else:
                        self.ewma_valid.append(ewma_coeff*self.scores_valid[-1] + (1 - ewma_coeff)*self.ewma_valid[-1])
                        
                    if self.scores_valid[-1] > min_score_valid and self.scores_valid[-1] == max(self.scores_valid) and saving:
                        with open('best_model_temp.pt', "wb") as f:
                            pickle.dump(self.model.state_dict(), f, pickle.HIGHEST_PROTOCOL)
                    
                    if early_stopping and (self.scores_valid[-1] > min_score_valid):
                        # Stops training early if EWMA of validation scores decreases
                        if self.ewma_valid[-1] < self.ewma_valid[-2]:
                            print("Training stopped early due to a decrease in validation performance")
                            self.convergence = "Validation score EWMA decrease"
                            break
                        # Stops training early if EWMA is not increasing significantly
                        elif ((self.ewma_valid[-1] - self.ewma_valid[-3])/(2*score_interval)*100) < min_slope:
                            print("Training stopped early due to near-zero validation performance improvement")
                            self.convergence = "Zero-slope validation score EWMA"
                
                # Checks for divergent behaviour
                if self.total_epochs > 5:
                    if self.losses_train[-1] > self.losses_train[0]:
                        print("Training stopped early due to divergent behaviour")
                        break
                
                    
            # Updates model parameters using gradient descent
            loss.backward()
            optimizer.step()
            
            # Outputs performance information
            print(' [-] epoch {:4}/{:}, train loss {:.6f} in {:.2f}s'.format(
                self.total_epochs, final_epochs, loss.item(), time.time()-start_time))
            if (i_epoch + 1) % score_interval != 0:
                self.epoch_times.append(time.time()-start_time)
            if (i_epoch + 1) % score_interval == 0:
                print("    Training score: " + str(round(self.scores_train[len(self.scores_train) - 1], 5)))
                if self.validation:
                    print("    Validation score: " + str(round(self.scores_valid[len(self.scores_valid) - 1], 5)))
        if self.validation and saving:
            with open('best_model_temp.pt', "rb") as f:
                self.model.load_state_dict(pickle.load(f))
    
    def predict(self, X, n_outputs = 5):
        # Generates predictions from X data - returns the indices of the n_outputs largest output nodes or raw predictions if n_outputs == -1

        training_before = self.model.training
        self.model.eval()
        
        # Converts and mounts data provided
        data_set = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
        X = data_set.tensors[0]
        X = X.to(self.device)
        with torch.no_grad():
            predictions = self.model(X)
            
        # Converts neural network predictions to numpy format
        numpy_predictions = predictions.clone().detach().numpy()
        
        # Get the most pertinent documents (n_outputs of them) or returns the raw output of the neural network if n_outputs == -1
        if n_outputs == -1:
            y_pred = numpy_predictions
        else:
            y_pred = list()
            for prediction in numpy_predictions:
                y_pred.append(np.argpartition(prediction, -n_outputs)[-n_outputs:])
            y_pred = np.asarray(y_pred)
            
        # Sets model back to training mode if it was previously in training mode
        if training_before:
            self.model.train()
            
        return y_pred
    
    def plot_ewma(self, filename = None):
        if self.validation:
            scores_train = np.asarray(self.scores_train)*100
            scores_valid = np.asarray(self.scores_valid)*100
            ewma_train = np.asarray(self.ewma_train)*100
            ewma_valid = np.asarray(self.ewma_valid)*100
            if filename != None:
                dpi = 400
            else: dpi = 150
            plt.figure(figsize = (4, 3), dpi = dpi)
            plt.plot(self.epochs_plot, scores_train, label = "Training", color = "blue")
            plt.plot(self.epochs_plot, ewma_train, label = "Training EWMA", color = "purple")
            plt.plot(self.epochs_plot, scores_valid, label = "Validation", color = "orange")
            plt.plot(self.epochs_plot, ewma_valid, label = "Validation EWMA", color = "red")
            plt.xlabel("Number of epochs")
            plt.ylabel("Score (%)")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol = 2, 
                           fancybox = True, shadow = False, fontsize = 8)
            plt.tight_layout()
            if filename != None:
                plt.savefig(filename)
    
    def plot(self, filename = None):
        if self.validation:
            scores_train = np.asarray(self.scores_train)*100
            scores_valid = np.asarray(self.scores_valid)*100
            if filename != None:
                dpi = 400
            else: dpi = 150
            plt.figure(figsize = (4, 3), dpi = dpi)
            plt.plot(self.epochs_plot, scores_train, label = "Training")
            plt.plot(self.epochs_plot, scores_valid, label = "Validation")
            plt.xlabel("Number of epochs")
            plt.ylabel("Score (%)")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol = 2, 
                           fancybox = True, shadow = False, fontsize = 8)
            plt.tight_layout()
            if filename != None:
                plt.savefig(filename)
            
    def quick_score(self, y, y_pred): 
        # Calculates scores when predicted y values are already known
        score = 0
        for idx, docs in enumerate(y):
            if len(np.intersect1d(np.asarray(docs), y_pred[idx, :])) > 0:
                score += 1
        score = score/len(y_pred)
        return score
                
    def score(self, X, y, n_outputs = 5):
        y_pred = self.predict(X, n_outputs)
        score = self.quick_score(y, y_pred)
        return score

