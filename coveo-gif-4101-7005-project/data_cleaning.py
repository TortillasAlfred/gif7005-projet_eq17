__author__ = "Francis Brochu"
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pds

def encode_one_hot(x):
    ldict = {}
    labels = []
    count = 0
    
    for el in x:
        if (el not in ldict) and (el == el):  # nan != nan
            ldict[el] = count
            labels.append(el)
            count += 1
    
    y = np.zeros(shape=(len(x),count), dtype=bool)
    for i, el in enumerate(x):
        if el == el:  # nan != nan
            y[i, ldict[el]] = 1
    
    return y, np.asarray(labels)

def apply_one_hot(x, labels):
    ldict = {}
    count = 0
    for el in labels:
        ldict[el] = count
        count += 1
    
    y = np.zeros(shape=(len(x), count))
    for i, el in enumerate(x):
        if el in ldict:
            y[i, ldict[el]] = 1
            
    return y


train_search = pds.read_csv("coveo_searches_train.csv")

search_features = ["search_id", "search_cause", "query_expression", "query_pipeline", "facet_title", "user_type"]

train_search = train_search[search_features]

cause_one_hot, cause_labels = encode_one_hot(train_search.search_cause.values)
query_one_hot, query_labels = encode_one_hot(train_search.query_pipeline.values)
facet_one_hot, facet_labels = encode_one_hot(train_search.facet_title.values)
user_one_hot, user_labels = encode_one_hot(train_search.user_type.values)

del_features = ["search_cause", "query_pipeline", "facet_title", "user_type"]
for el in del_features:
    del train_search[el]

data = np.hstack((train_search.values, cause_one_hot, query_one_hot, facet_one_hot, user_one_hot))

labels = np.hstack((train_search.columns.values, cause_labels, query_labels, facet_labels, user_labels))

np.save("train_searches.npy", data)
np.save("searches_features.npy", labels)

valid_search = pds.read_csv("coveo_searches_valid.csv")

valid_search = valid_search[search_features]

cause_one_hot = apply_one_hot(valid_search.search_cause.values, cause_labels)
query_one_hot = apply_one_hot(valid_search.query_pipeline.values, query_labels)
facet_one_hot = apply_one_hot(valid_search.facet_title.values, facet_labels)
user_one_hot = apply_one_hot(valid_search.user_type.values, user_labels)

for el in del_features:
    del valid_search[el]

data = np.hstack((valid_search.values, cause_one_hot, query_one_hot, facet_one_hot, user_one_hot))

np.save("valid_searches.npy", data)

test_searches = pds.read_csv("coveo_searches_test.csv")

test_searches = test_searches[search_features]

cause_one_hot = apply_one_hot(test_searches.search_cause.values, cause_labels)
query_one_hot = apply_one_hot(test_searches.query_pipeline.values, query_labels)
facet_one_hot = apply_one_hot(test_searches.facet_title.values, facet_labels)
user_one_hot = apply_one_hot(test_searches.user_type.values, user_labels)

for el in del_features:
    del test_searches[el]

data = np.hstack((test_searches.values, cause_one_hot, query_one_hot, facet_one_hot, user_one_hot))

np.save("test_searches.npy", data)

### Clicks and documents

train_clicks = pds.read_csv("coveo_clicks_train.csv")

click_features = ["document_source", "document_title", "document_id"]

# click correspondances between searches and documents

sc = train_clicks[["search_id", "document_id"]]

np.save("train_correspondance.npy", sc.values)

train_clicks = train_clicks[click_features]

train_clicks.drop_duplicates(subset="document_id", inplace=True)

train_ids = train_clicks.document_id.values

np.save("train_documents.npy", train_clicks.values)
np.save("document_labels.npy", train_clicks.columns.values)

valid_clicks = pds.read_csv("coveo_clicks_valid.csv")

sc = valid_clicks[["search_id", "document_id"]]

np.save("valid_correspondance.npy", sc.values)

