# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:52:44 2018

@author: David
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:17:16 2018

@author: David
"""

import time

import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
# Sklearn libraries

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from torch.utils.data import RandomSampler
import torch.utils.data
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
    X_train_orig_bow, X_valid_orig_bow, X_test_orig_bow, y_train_orig, y_valid_orig, _, all_docs_ids = loader_filtered_bow.load_transform_data()
    
    loader_unfiltered_bow = DataLoader(vectorizer=vect_bow, one_hot_encoder=enc, 
                                 search_features=DataLoader.default_search_features,
                                 click_features=DataLoader.default_click_features,
                                 data_folder_path="./data/", numpy_folder_path="./data/bow_oh_unfiltered/", 
                                 load_from_numpy=False, filter_no_clicks=False)
    
    X_train_orig_bow_unf, X_valid_orig_bow_unf, X_test_orig_bow_unf, y_train_orig_unf, y_valid_orig_unf, _, all_docs_ids_unf = loader_unfiltered_bow.load_transform_data()
    
    vect_we = 
    enc = OneHotEncoder()
    
    loader_filtered_we = DataLoader(vectorizer=vect_we, one_hot_encoder=enc, 
                                 search_features=DataLoader.default_search_features,
                                 click_features=DataLoader.default_click_features,
                                 data_folder_path="./data/", numpy_folder_path="./data/we_filtered/", 
                                 load_from_numpy=True, filter_no_clicks=True)

    X_train_orig_we, X_valid_orig_we, X_test_orig_we, y_train_orig, y_valid_orig, _, all_docs_ids = loader_filtered_we.load_transform_data()
    

    loader_unfiltered_we = DataLoader(vectorizer=vect_we, one_hot_encoder=enc, 
                                 search_features=DataLoader.default_search_features,
                                 click_features=DataLoader.default_click_features,
                                 data_folder_path="./data/", numpy_folder_path="./data/we_unfiltered/", 
                                 load_from_numpy=True, filter_no_clicks=False)

    X_train_orig_we_unf, X_valid_orig_we_unf, X_test_orig_we_unf, y_train_orig_unf, y_valid_orig_unf, _, all_docs_ids_unf = loader_unfiltered_we.load_transform_data()

    total_train_docs = len(np.unique(np.concatenate(y_train_orig)))
    y_train = np.concatenate(y_train_orig)
    y_valid = np.concatenate(y_valid_orig)
 
    ids_train = all_docs_ids[y_train]
    ids_valid = all_docs_ids[y_valid]

    # Converts bag of words or word embeddings to integers
    X_train_orig_bow = X_train_orig_bow.astype(float)
    X_valid_orig_bow = X_valid_orig_bow.astype(float)
    X_valid_orig_bow_unf = X_valid_orig_bow_unf.astype(float)
    X_test_orig_bow = X_test_orig_bow.astype(float)
    
    X_train_orig_we = X_train_orig_we.astype(float)
    X_valid_orig_we = X_valid_orig_we.astype(float)
    X_valid_orig_we_unf = X_valid_orig_we_unf.astype(float)
    X_test_orig_we = X_test_orig_we.astype(float)
    
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
    np.argsort(train_freq)
    sum(train_freq)
    weights = 1/train_freq
    sum(train_freq == 1)
    sum(train_freq == 2)
    sum(train_freq == 3)
    np.max(train_freq)
    sum(train_freq[np.argpartition(train_freq,-57)[-57:]])/sum(train_freq)
    
    len(np.concatenate(y_train_orig))
    # DOE Functions  =======================================================================================================================================
    
    def save_model(model, name):
        model_name = name + ".file"
        plot_name = name + " - Plot.png"
        ewma_name = name + " - EWMA Plot.png"
        with open(model_name, "wb") as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        model.plot(plot_name)
        model.plot_ewma(ewma_name)

    def doe_stats(model, x_type = "bow"):
        opt_index = np.where(model.scores_valid == np.max(model.scores_valid))[0][0] # Finds index of optimal performance
        print("Max validation score: " + str(model.scores_valid[opt_index]))
        print("Associated training score: " + str(model.scores_train[opt_index]))
        if x_type == "bow":
            print("Top 200 score: " + str(model.score(X_valid_orig_bow, y_valid_orig, n_outputs = 200)))
        elif x_type == "we":
            print("Top 200 score: " + str(model.score(X_valid_orig_we, y_valid_orig, n_outputs = 200)))
        print("Optimal epoch: " + str(model.epochs_plot[opt_index]))
        if len(model.epoch_times) > 0:
            print("Average epoch time: " + str(np.mean(model.epoch_times)))
        print("Total epochs: " + str(model.total_epochs))

    def get_experiment_stats(exp_nums, prefix, x_type = "bow", filename = None):
        if filename == None:
            for num in exp_nums:
                with open(prefix + " " + str(num) + ".file", "rb") as f:
                    model = pickle.load(f)
                print("Experiment " + str(num) + " ====================================")
                doe_stats(model, x_type)
                print("")
            
    
    def load_model(model_name):
        filename = model_name + ".file"
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
        doe_stats(doe_model, x_type = vect_type)
        
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
        doe_stats(doe_model, x_type = vect_type)
        
        
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
  
    class BinaryContourModel():
        def __init__(self):
            self.model= LinearRegression()
        
        def transform(self, xn, xc):
            xn = np.asarray(xn).astype(float)[:, np.newaxis]
            xc = np.asarray(xc).astype(float)[:, np.newaxis]
            X = np.concatenate((xn, xc, xn*xc), axis = 1)
            return X
        
        def fit(self, xn, xc, error):
            self.model.fit(self.transform(xn, xc), error)
            self.R2 = r2_score(error, self.predict(xn, xc))
        
        def predict(self, xn, xc):
            return self.model.predict(self.transform(xn, xc))
        
        def get_max(self, xc = None):
            b = self.model.coef_
            if xc == None:
                xn_opt = -b[1]/b[2]
                xc_opt = -b[0]/b[2]
                error_opt = self.predict([xn_opt], [xc_opt])
                print("Optimum number of hidden layers: " + str(xc_opt))
                print("Optimum number of neurons: " + str(xn_opt))
                print("Optimum error: " + str(error_opt))
            else:
                xn_opt = -b[1]/b[2]
                error_opt = self.predict([xn_opt], [xc])
                print(str(xc) + " hidden layers")
                print("Optimum number of neurons: " + str(xn_opt))
                print("Optimum error: " + str(error_opt))
                
    def save_test_predictions(model_name, x_type):
        model = load_model(model_name)
        if x_type == "bow":
            X_test = X_test_orig_bow
        elif x_type == "we":
            X_test = X_test_orig_we
        
        # Creates a list of numpy arrays containing the top 5 predicted documents for each search, in descending order of relevance
        predictions = model.predict(X_test, n_outputs = -1)
        top_5 = []
        for row in predictions:
            top_5.append(np.flip(np.argsort(row)[-5:]))
        
        # Imports search ids for the testing data
        s_test = pd.read_csv("data/coveo_searches_test.csv")
        search_ids = s_test.loc[:, "search_id"]
        final_list = []
        
        # Creates a new list appending search ids to the left of the top 5 documents
        for idx, docs in enumerate(top_5):
            temp_list = []
            temp_list.append(search_ids[idx])
            for doc in docs:
                temp_list.append(all_docs_ids[doc])
            final_list.append(temp_list)
        
        test_predictions = pd.DataFrame(final_list)
        test_predictions.to_csv("test predictions - " + x_type + " neural network.csv", index=False, header=False)
        
    def save_valid_predictions(model_name, x_type):
        model = load_model(model_name)
        if x_type == "bow":
            X_valid = X_valid_orig_bow_unf
        elif x_type == "we":
            X_valid = X_valid_orig_we_unf
        
        # Creates a list of numpy arrays containing the top 5 predicted documents for each search, in descending order of relevance
        predictions = model.predict(X_valid, n_outputs = -1)
        top_5 = []
        for row in predictions:
            top_5.append(np.flip(np.argsort(row)[-5:]))
        
        # Imports search ids for the testing data
        s_test = pd.read_csv("data/coveo_searches_valid.csv")
        search_ids = s_test.loc[:, "search_id"]
        final_list = []
        
        # Creates a new list appending search ids to the left of the top 5 documents
        for idx, docs in enumerate(top_5):
            temp_list = []
            temp_list.append(search_ids[idx])
            for doc in docs:
                temp_list.append(all_docs_ids[doc])
            final_list.append(temp_list)
        
        test_predictions = pd.DataFrame(final_list)
        test_predictions.to_csv("validation predictions - " + x_type + " neural network.csv", index=False, header=False)
    

    # Bag of Words DOE =====================================================================================================================================

    # Group 1 =================================================================
    
    run_doe("BOW - OPT 2", n_neurons = 4947, n_layers = 2, vect_type = "bow", drop_rate = 0.3, score_interval = 1, ewma_coeff = 0.2,
            min_score_valid = 0.45, early_stopping = False, n_epochs = 200)

    
    
    
    


