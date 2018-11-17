# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:40:49 2018

@author: David
"""

import pandas
import numpy

# Imports the raw data
s_train = pandas.read_csv("coveo_searches_train.csv")
c_train = pandas.read_csv("coveo_clicks_train.csv")
s_val = pandas.read_csv("coveo_searches_valid.csv")
c_val = pandas.read_csv("coveo_clicks_valid.csv")
s_test = pandas.read_csv("coveo_searches_test.csv")

# Defines features of interest to retain
s_features = ["search_id", "search_datetime",  "search_cause", "query_expression", "user_device", "user_country",  "user_city", "user_region"]
c_features = ["search_id", "document_id", "document_source", "click_rank"]

# Removes the unnecessary features
s_train = s_train.loc[:, s_features]
c_train = c_train.loc[:, c_features]
s_val = s_val.loc[:, s_features]
c_val = c_val.loc[:, c_features]
s_test = s_test.loc[:, s_features]

# Gets classes (unique documents) and sources
classes = pandas.unique(c_train.loc[:, "document_id"])
sources = pandas.unique(c_train.loc[:, "document_source"])

# Associates training and validation set searches with the documents that were clicked on
docs_train, docs_val = list(), list()
for s_id in s_train.loc[:, "search_id"]:
    docs_train.append(pandas.unique(c_train.loc[c_train.loc[:, "search_id"] == s_id, "document_id"]))

for s_id in s_val.loc[:, "search_id"]:
    docs_val.append(pandas.unique(c_val.loc[c_train.loc[:, "search_id"] == s_id, "document_id"]))

r_train = docs_train # List containing the documents clicked on for each search in the training set
r_val = docs_val # List containing the documents clicked on for each search in the validation set

num_clicks_train = numpy.repeat(0, len(docs_train))
for i in range(len(docs_train)):
    num_clicks_train[i] = len(docs_train[i])


num_clicks_val = numpy.repeat(0, len(docs_val))
for i in range(len(docs_val)):
    num_clicks_val[i] = len(docs_val[i])
docs_val[2]
sum(num_clicks_train != 0)
sum(num_clicks_val != 0)





