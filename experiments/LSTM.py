import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from loading.wordVectorizer import *
from loading.dataLoader import DataLoader
from loading.queryDocBatchDataLoader import QueryDocBatchDataLoader
from loading.oneHotEncoder import OneHotEncoder

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)



class LSTMInfer(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, nb_queries, nb_class):
        super(LSTMInfer, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(nb_queries, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, nb_class)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        response_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        response_scores = F.log_softmax(response_space, dim=1)
        return response_scores


vectWV = DictSentenceVectorizerSpacy()
enc = OneHotEncoder()
loader = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                    search_features=DataLoader.default_search_features,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="../data/", numpy_folder_path="../data/qd_wv/",
                                    load_from_numpy=True, filter_no_clicks=False)
'''loader = QueryDocBatchDataLoader(vectorizer=vectWV, encoder=enc, batch_size=4e4, data_folder_path="../data/",
                                 numpy_folder_path="../data/qd_wv/", load_from_numpy=False,
                                 filter_no_clicks=True, load_dummy=False, generate_pairs=False)'''

loader.load_transform_data()
searches_train = loader.load_all_from_numpy("X_train")
clicks_train = loader.load_all_from_numpy("y_train")
training_data = loader.load_all_from_numpy("test_train")
documents_train = loader.load_all_from_numpy("all_docs")


nb_exemples = training_data.shape[0]
nb_doc = documents_train.shape[0]
nb_features = training_data[0][0].shape[0]


'''lstm = nn.LSTM(20, 20)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 20) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 20),
          torch.randn(1, 1, 20))

inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 20), torch.randn(1, 1, 20))  # clean out hidden state
out, hidden = lstm(inputs, hidden)'''

model = LSTMInfer(nb_features, nb_features, nb_exemples, nb_doc)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    inputs = torch.tensor(training_data[0,0], dtype=torch.long)
    doc_score = model(inputs)
    print(doc_score)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for query, document in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        query_in = torch.tensor(query)
        docs = [torch.tensor[doc] for doc in document]

        # Step 3. Run our forward pass.
        doc_score = model(query_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(doc_score, docs)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = training_data[0][0]
    doc_score = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(doc_score)
