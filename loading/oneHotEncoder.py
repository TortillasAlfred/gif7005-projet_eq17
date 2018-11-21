import numpy as np

class OneHotEncoder:
    def fit_transform(self, data_train, *data):
        self.fit(data_train)

        transformed_data = []
        transformed_data.append(self.transform(data_train))

        for d in data:
            transformed_data.append(self.transform(d))

        return transformed_data

    def fit(self, data_train):
        ldict = {}
        labels = []
        count = 0
        
        for el in data_train:
            if (el not in ldict.keys()) and (el == el):  # nan != nan
                ldict[el] = count
                labels.append(el)
        
        self.labels = np.asarray(labels)

    def transform(self, data):
        ldict = {}
        count = 0
        for el in self.labels:
            ldict[el] = count
            count += 1
        
        y = np.zeros(shape=(len(data), count), dtype=bool)
        for i, el in enumerate(data):
            if el in ldict.keys():
                y[i, ldict[el]] = True
                
        return y