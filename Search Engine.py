# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:17:16 2018

@author: David
"""

import time

import numpy as np
from matplotlib import pyplot as plt
import pickle
# Sklearn libraries

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from torch.utils.data import RandomSampler
import collections
from torch.autograd import Variable

# Project libraries
from loading.dataLoader import DataLoader
from loading.oneHotEncoder import OneHotEncoder
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.wordVectorizer import WordVectorizer
from scorers.coveo_scorer import coveo_score
from neuralnetworks import NeuralNetwork, SearchEngine, model_stats

if __name__ == '__main__':

    # Loads the training and validation data sets
    vect_bow = BagOfWordsVectorizer()
    enc = OneHotEncoder()
    loader_filtered_bow = DataLoader(vectorizer=vect_bow, one_hot_encoder=enc, 
                                 search_features=DataLoader.default_search_features,
                                 click_features=DataLoader.default_click_features,
                                 data_folder_path="./data/", numpy_folder_path="./data/bow_oh_filtered/", 
                                 load_from_numpy=False, filter_no_clicks=True)
    X_train_orig_bow, X_valid_orig_bow, _, y_train_orig, y_valid_orig, _, all_docs_ids = loader_filtered_bow.load_transform_data()
    
    vect_we = WordVectorizer()
    enc = OneHotEncoder()
    loader_filtered_we = DataLoader(vectorizer=vect_we, one_hot_encoder=enc, 
                                 search_features=DataLoader.default_search_features,
                                 click_features=DataLoader.default_click_features,
                                 data_folder_path="./data/", numpy_folder_path="./data/we_filtered/", 
                                 load_from_numpy=False, filter_no_clicks=True)

    X_train_orig_we, X_valid_orig_we, _, y_train_orig, y_valid_orig, _, all_docs_ids = loader_filtered_we.load_transform_data()
    
    total_train_docs = len(np.unique(np.concatenate(y_train_orig)))
    y_train = np.concatenate(y_train_orig)
    y_valid = np.concatenate(y_valid_orig)
 
    ids_train = all_docs_ids[y_train]
    ids_valid = all_docs_ids[y_valid]

    # Converts bag of words or word embeddings to integers
    X_train_orig_bow = X_train_orig_bow.astype(float)
    X_valid_orig_bow = X_valid_orig_bow.astype(float)
    X_train_orig_we = X_train_orig_we.astype(float)
    X_valid_orig_we = X_valid_orig_we.astype(float)
    
    X_train_we = X_train_orig_we.astype(float)
    X_valid_we = X_valid_orig_we.astype(float)
    
    
    X_train_bow = X_train_orig_bow.astype(float)
    X_valid_bow = X_valid_orig_bow.astype(float)
    
    X_train_we = X_train_orig_we.astype(float)
    X_valid_we = X_valid_orig_we.astype(float)
    
    # Expands data matrices to match expanded target vectors
    repeats_train = []
    for i in y_train_orig:
        repeats_train.append(len(i))
    repeats_valid = []
    for i in y_valid_orig:
        repeats_valid.append(len(i))
        
    def convert_X(X_orig, y_new, repeats):
        X_new = np.zeros(shape = (len(y_new), np.shape(X_orig)[1]))
        j = 0
        for i in range(np.shape(X_orig)[0]):
            X_new[j, :] = X_orig[i, :]
            if repeats[i] > 1:
                for idx in range(repeats[i] - 1):
                    j += 1
                    X_new[j, :] = X_orig[i, :]
            j += 1
        return X_new
    X_train_bow = convert_X(X_train_bow, y_train, repeats_train)
    X_train_we = convert_X(X_train_we, y_train, repeats_train)
    
    X_valid_bow = convert_X(X_valid_bow, y_valid, repeats_valid)
    X_valid_we = convert_X(X_valid_we, y_valid, repeats_valid)
    

    # Calculates weightings of classes so that frequent documents do not dominate the gradient descent
    train_unique, train_freq = np.unique(np.concatenate(y_train_orig), return_counts = True)
    weights = 1/train_freq
    
    # DOE Functions  =======================================================================================================================================
    
    def save_model(model, name):
        model_name = name + ".file"
        plot_name = name + " - Plot.png"
        ewma_name = name + " - EWMA Plot.png"
        with open(model_name, "wb") as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        model.plot(plot_name)
        model.plot_ewma(ewma_name)

    def doe_stats(model):
        opt_index = np.where(model.scores_valid == np.max(model.scores_valid))[0][0] # Finds index of optimal performance
        print("Max validation score: " + str(model.scores_valid[opt_index]))
        print("Associated training score: " + str(model.scores_train[opt_index]))
        print("Top 200 score: " + str(model.score(X_valid_orig, y_valid_orig, n_outputs = 200)))
        print("Optimal epoch: " + str(model.epochs_plot[opt_index]))
        if len(model.epoch_times) > 0:
            print("Average epoch time: " + str(np.mean(model.epoch_times)))
        print("Total epochs: " + str(model.total_epochs))

    def get_experiment_stats(exp_nums, prefix, filename = None):
        if filename == None:
            for num in exp_nums:
                with open(prefix + " " + str(num) + ".file", "rb") as f:
                    model = pickle.load(f)
                print("Experiment " + str(num) + " ====================================")
                doe_stats(model)
                print("")
            
    
    def load_model(exp_id, x_type):
        filename = x_type + " - DOE " + exp_id + ".file"
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model
    
    def run_doe(exp_name, n_neurons, n_layers, vect_type = "bow", score_interval = 5, device = "cpu", n_epochs = 10000, optimizer = "sgd", 
                learning_rate = 0.1, momentum = 0.9, class_weights = [], early_stopping = True, ewma_coeff = 0.4, 
                min_slope = 0.001, min_score_valid = 0.13, saving = True, activation_function = "relu", drop_rate = 0):
        # Builds a neural network with the supplied parameters and displays the performance results
        
        # Creates network architecture
        size = int(round(n_neurons/n_layers))
        hidden_layer_sizes = [size]*n_layers
        activation_functions = [activation_function]*n_layers
        drop_rates = [drop_rate]*(n_layers + 1)
        if drop_rate != 0:
            drop_rates[0] = 0.2
            drop_rates[-1] = 0

        # Selects training and validation data
        if vect_type == "bow":
            X_train = X_train_bow
            X_train_orig = X_train_orig_bow
            X_valid_orig = X_valid_orig_bow
        elif vect_type == "we":
            X_train = X_train_we
            X_train_orig = X_train_orig_we
            X_valid_orig = X_valid_orig_we
        else:
            print("Received unrecognized word vector type")
        
        # Initializes the neural network
        doe_model = SearchEngine(in_features = X_train.shape[1], out_features = len(np.unique(y_train)), 
                                hidden_layer_sizes = hidden_layer_sizes, activation_functions = activation_functions, drop_rates = drop_rates)
        
        # Fits the neural network.  Training is stopped early by default using an EWMA of the validation performance
        doe_model.fit(X_train, y_train, X_train_orig, y_train_orig, X_valid_orig, y_valid_orig, score_interval = score_interval,
                  device = device, n_epochs = n_epochs, optimizer = optimizer, learning_rate = learning_rate, momentum = momentum, weight = class_weights, 
                  early_stopping = early_stopping, ewma_coeff = ewma_coeff, min_slope = min_slope, min_score_valid = min_score_valid, saving = saving)
        
        # Saves the model and prints performance results
        save_model(doe_model, exp_name)
        doe_stats(doe_model)
        
    def train_doe(exp_id, vect_type = "bow", score_interval = 5, device = "cpu", n_epochs = 10000, optimizer = "sgd", 
                learning_rate = 0.1, momentum = 0.9, class_weights = [], early_stopping = True, ewma_coeff = 0.4, 
                min_slope = 0.001, min_score_valid = 0.13):
        
        if vect_type == "bow":
            X_train = X_train_bow
            X_train_orig = X_train_orig_bow
            X_valid_orig = X_valid_orig_bow
            exp_name = "BOW - DOE " + str(exp_id)
        elif vect_type == "we":
            X_train = X_train_we
            X_train_orig = X_train_orig_we
            X_valid_orig = X_valid_orig_we
            exp_name = "WE - DOE " + str(exp_id)
        else:
            print("Received unrecognized word vector type")
        
        doe_model = load_model(exp_id, vect_type)
        
        doe_model.fit(X_train, y_train, X_train_orig, y_train_orig, X_valid_orig, y_valid_orig, score_interval = score_interval,
                  device = device, n_epochs = n_epochs, optimizer = optimizer, learning_rate = learning_rate, momentum = momentum, weight = class_weights, 
                  early_stopping = early_stopping, ewma_coeff = ewma_coeff, min_slope = min_slope, min_score_valid = min_score_valid)
        save_model(doe_model, exp_name)
        doe_stats(doe_model)
    # Bag of Words DOE =====================================================================================================================================
    # Group 1 =================================================================
    """
    run_doe("BOW - DOE 1", n_neurons = 500, n_layers = 1, vect_type = "bow")
    run_doe("BOW - DOE 2", n_neurons = 2500, n_layers = 1, vect_type = "bow")
    run_doe("BOW - DOE 3", n_neurons = 500, n_layers = 3, vect_type = "bow")
    run_doe("BOW - DOE 4", n_neurons = 2500, n_layers = 3, vect_type = "bow")
    run_doe("BOW - DOE 5", n_neurons = 1500, n_layers = 2, vect_type = "bow")
    
    get_experiment_stats([1, 2, 3, 4, 5], x_type = "BOW")
    
    # Group 2 =================================================================
    
    run_doe("BOW - DOE 6", n_neurons = 1500, n_layers = 1, vect_type = "bow")
    run_doe("BOW - DOE 7", n_neurons = 100, n_layers = 2, vect_type = "bow")
    run_doe("BOW - DOE 8", n_neurons = 3000, n_layers = 2, vect_type = "bow")
    run_doe("BOW - DOE 9", n_neurons = 1500, n_layers = 4, vect_type = "bow")
    
    get_experiment_stats([6, 7, 8, 9], x_type = "BOW")
    
    # Group 3 =================================================================
    
    # Initial group 3 experiments
    run_doe("BOW - DOE 10A", n_neurons = 5000, n_layers = 3, vect_type = "bow")
    run_doe("BOW - DOE 11A", n_neurons = 5500, n_layers = 4, vect_type = "bow")
    run_doe("BOW - DOE 12A", n_neurons = 5500, n_layers = 2, vect_type = "bow")
    run_doe("BOW - DOE 13A", n_neurons = 4500, n_layers = 4, vect_type = "bow")
    run_doe("BOW - DOE 14A", n_neurons = 4500, n_layers = 2, vect_type = "bow")
    
    # Repeats done with dropout added to improve validation performance
    run_doe("BOW - DOE 10B", n_neurons = 5000, n_layers = 3, vect_type = "bow", drop_rate = 0.3)
    run_doe("BOW - DOE 11B", n_neurons = 5500, n_layers = 4, vect_type = "bow", drop_rate = 0.3)
    run_doe("BOW - DOE 12B", n_neurons = 5500, n_layers = 2, vect_type = "bow", drop_rate = 0.3)
    run_doe("BOW - DOE 13B", n_neurons = 4500, n_layers = 4, vect_type = "bow", drop_rate = 0.3)
    run_doe("BOW - DOE 14B", n_neurons = 4500, n_layers = 2, vect_type = "bow", drop_rate = 0.3)
    
    get_experiment_stats(["10A", "10B"], x_type = "BOW")
    get_experiment_stats(["11A", "11B"], x_type = "BOW")
    get_experiment_stats(["12A", "12B"], x_type = "BOW")
    get_experiment_stats(["13A", "13B"], x_type = "BOW")
    get_experiment_stats(["14A", "14B"], x_type = "BOW")

    # Group 4 =================================================================
    run_doe("BOW - DOE 15", n_neurons = 5000, n_layers = 5, vect_type = "bow", drop_rate = 0.3)
    run_doe("BOW - DOE 16", n_neurons = 4000, n_layers = 3, vect_type = "bow", drop_rate = 0.3)
    run_doe("BOW - DOE 17", n_neurons = 6000, n_layers = 3, vect_type = "bow", drop_rate = 0.3)
    run_doe("BOW - DOE 18", n_neurons = 5000, n_layers = 1, vect_type = "bow", drop_rate = 0.3, learning_rate = 1)
    
    get_experiment_stats([15, 16, 17, 18], x_type = "BOW")
    
    # Optimization group ======================================================
    
    run_doe("BOW - OPT 1", n_neurons = 4947, n_layers = 2, vect_type = "bow", drop_rate = 0.3, score_interval = 1, ewma_coeff = 0.2,
            min_score_valid = 0.45, early_stopping = False, n_epochs = 250)
    run_doe("BOW - OPT 2", n_neurons = 4947, n_layers = 2, vect_type = "bow", drop_rate = 0.3, score_interval = 1, ewma_coeff = 0.2,
            min_score_valid = 0.01, early_stopping = False, n_epochs = 250)
    run_doe("BOW - OPT 3", n_neurons = 4947, n_layers = 2, vect_type = "bow", drop_rate = 0.3, score_interval = 1, ewma_coeff = 0.2,
            min_score_valid = 0.45, early_stopping = False, n_epochs = 250)
    run_doe("BOW - OPT 4", n_neurons = 4947, n_layers = 2, vect_type = "bow", drop_rate = 0.3, score_interval = 1, ewma_coeff = 0.2,
            min_score_valid = 0.45, early_stopping = False, n_epochs = 250)
    run_doe("BOW - OPT 5", n_neurons = 4947, n_layers = 2, vect_type = "bow", drop_rate = 0.3, score_interval = 1, ewma_coeff = 0.2,
            min_score_valid = 0.45, early_stopping = False, n_epochs = 250)

    get_experiment_stats([1], prefix = "BOW - OPT")
    # Optimization group ======================================================


    # Response surface
    
    # Selects which experiments to include in the response surface model
    exp_start = 10
    exp_finish = 18
  
    # All data
    xn = [500, 2500, 500, 2500, 1500, 
          1500, 100, 3000, 1500, 
          5000, 5500, 5500, 4500, 4500, 
          5000, 4000, 6000, 5000,
          4947]
    xc = [1, 1, 3, 3, 2, 
          1, 2, 2, 4,
          3, 4, 2, 4, 2, 
          5, 3, 3, 2, 
          2]
    error = [0.391, 0.395, 0.408, 0.449, 0.444, 
         0.392, 0.413, 0.479, 0.388, 
         0.490, 0.484, 0.492, 0.482, 0.495, 
         0.470, 0.481, 0.487, 0.491,
         0.493]
    
    class ContourModel():
        def __init__(self):
            self.model= LinearRegression()
        
        def transform(self, xn, xc):
            xn = np.asarray(xn).astype(float)[:, np.newaxis]
            xc = np.asarray(xc).astype(float)[:, np.newaxis]
            X = np.concatenate((xn, xc, xn*xc, xn**2, xc**2), axis = 1)
            return X
        
        def fit(self, xn, xc, error):
            self.model.fit(self.transform(xn, xc), error)
            self.R2 = r2_score(error, self.predict(xn, xc))
        
        def predict(self, xn, xc):
            return self.model.predict(self.transform(xn, xc))
        
        def get_max(self, xc = 2):
            b = self.model.coef_
            xn_opt = (-b[0] - b[2]*xc)/(2*b[3])
            error_opt = self.predict([xn_opt], [xc])
            print(str(xc) + " hidden layers")
            print("Optimum number of neurons: " + str(xn_opt))
            print("Optimum error: " + str(error_opt))

    cont_model = ContourModel()
    cont_model.fit(xn[(exp_start - 1):exp_finish], xc[(exp_start - 1):exp_finish], error[(exp_start - 1):exp_finish])
    
    cont_model.get_max(xc = 2)
    
    res = 500
    h_x = 1000/res
    h_y = 1/res
    x = xn
    y = xc
    x_min = -200
    y_min = 0
    x_max = max(x) + 3000
    y_max = max(y) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))
    marker_size = 12
    font_name = "Times New Roman"
    font_size = 8
    #X_mod = np.column_stack((xx.ravel(), yy.ravel()))
    #Z = list()
    #X_mod = np.concatenate((X_mod, (X_mod[:, 0]*X_mod[:, 1])[:, np.newaxis], X_mod[:, 0][:, np.newaxis]**2, 
    #                        X_mod[:, 1][:, np.newaxis]**2), axis = 1)
    
    Z = cont_model.predict(xx.ravel(), yy.ravel())
    #Z = cont_model.predict(X_mod)
    #for combo in X_mod:
    #    Z.append(cont_model.predict(np.array([combo[0], combo[1], combo[0]*combo[1], combo[0]**2, combo[1]**2])[np.newaxis, :]))
    #Z = np.asarray(Z)
    Z = Z.reshape(xx.shape)
    cont_model.model.get_params()
    group_1 = [1, 2, 3, 4, 5]
    group_2 = [6, 7, 8, 9]
    group_3 = [10, 11, 12, 13, 14]
    group_4 = [15, 16, 17, 18]
    group_5 = [19]

    label_groups = [group_1, group_2, group_3, group_4, group_5]
    labels = ["G1", "G2", "G3", "G4", "Optimum"]

    colours = ["blue", "orange", "red", "green", "magenta"]
    plt.figure(figsize = (4, 3), dpi = 1000)
    contours = plt.contourf(xx, yy, Z, 8)
    cbar = plt.colorbar(contours)
    cbar.ax.tick_params(labelsize= font_size)
    plt.xlabel("Nombre de neurones", fontsize = font_size)
    plt.ylabel("Nombre de couches cachées", fontsize = font_size)
    yint = range(0, int(y_max)+1)
    plt.yticks(yint)
    plt.tick_params(axis='both', which='major', labelsize = (font_size - 1))
    i = 0
    for group_num, group in enumerate(label_groups):
        for idx, exp_num in enumerate(group):
            if group_num != (len(label_groups) - 1):
                if idx == 0:
                    plt.scatter(xn[i], xc[i], color = colours[group_num], s = marker_size, 
                                label = labels[group_num])
                else:
                    plt.scatter(xn[i], xc[i], color = colours[group_num], s = marker_size)
                plt.annotate(str(exp_num), (xn[i] - 200, xc[i] + 0.09), fontsize = (font_size - 1))
            else:
                if idx == 0:
                    plt.scatter(xn[i], xc[i], color = colours[group_num], marker = '*', s = marker_size, 
                                    label = labels[group_num])
                else:
                    plt.scatter(xn[i], xc[i], color = colours[group_num], marker = '*', s = marker_size)
            i += 1
    plt.legend(bbox_to_anchor = (0.50, -0.2), loc = 'upper center', ncol = len(labels), 
                            fancybox = False, shadow = False, fontsize = (font_size - 1))
    
    plt.tight_layout()
    plt.savefig("BOW - Response Surface.png")

    # Word Embeddings DOE =====================================================================================================================================
    # Group 1 =================================================================
    
    run_doe("WE - DOE 1", n_neurons = 500, n_layers = 3, vect_type = "we", drop_rate = 0.3)
    run_doe("WE - DOE 2", n_neurons = 2500, n_layers = 3, vect_type = "we", drop_rate = 0.3)
    run_doe("WE - DOE 3", n_neurons = 1500, n_layers = 2, vect_type = "we", drop_rate = 0.3)
    run_doe("WE - DOE 4", n_neurons = 500, n_layers = 1, vect_type = "we", drop_rate = 0.3, learning_rate = 1)
    run_doe("WE - DOE 5", n_neurons = 2500, n_layers = 1, vect_type = "we", drop_rate = 0.3, learning_rate = 1)
    
    
    
    run_doe("BOW - DOE 2", n_neurons = 2500, n_layers = 1, vect_type = "bow")
    run_doe("BOW - DOE 3", n_neurons = 500, n_layers = 3, vect_type = "bow")
    run_doe("BOW - DOE 4", n_neurons = 2500, n_layers = 3, vect_type = "bow")
    run_doe("BOW - DOE 5", n_neurons = 1500, n_layers = 2, vect_type = "bow")
    
    get_experiment_stats([1, 2, 3, 4, 5], prefix = "WE - DOE")

    """




