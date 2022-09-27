import numpy as np
import pickle
from numpy import log, dot, e
from numpy.random import rand
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns


class NaiveBayesClassifier():

    def calc_prior(self, features, target):
        self.prior = (features.groupby(target).apply(lambda x: len(x)) / self.rows).to_numpy()

        return self.prior
    
    def calc_statistics(self, features, target):
        
        self.mean = features.groupby(target).apply(np.mean).to_numpy()
        self.var = features.groupby(target).apply(np.var).to_numpy()
              
        return self.mean, self.var
    
    def gaussian_density(self, class_idx, x):     
        
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp((-1/2)*((x-mean)**2) / (2 * var))

        denominator = np.sqrt(2 * np.pi * var)
        prob = numerator / denominator
        return prob
    
    def calc_posterior(self, x):
        posteriors = []

        for i in range(self.count):
            conditional = np.sum(np.log(self.gaussian_density(i, x))) 
            posterior = self.prior + conditional
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]
     

    def fit(self, features, target):
        self.classes = np.unique(target)
        self.count = len(self.classes)
        self.feature_nums = features.shape[1]
        self.rows = features.shape[0]
        
        self.calc_statistics(features, target)
        self.calc_prior(features, target)
        
    def predict(self, features):
        preds = [self.calc_posterior(f) for f in features.to_numpy()]
        return preds

    def accuracy(self, y_test, y_pred):
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        return accuracy

    def visualize(self, y_true, y_pred, target):
        
        tr = pd.DataFrame(data=y_true, columns=[target])
        pr = pd.DataFrame(data=y_pred, columns=[target])
        
        
        fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(15,6))
        
        sns.countplot(x=target, data=tr, ax=ax[0], palette='viridis', alpha=0.7, hue=target, dodge=False)
        sns.countplot(x=target, data=pr, ax=ax[1], palette='viridis', alpha=0.7, hue=target, dodge=False)
        

        fig.suptitle('True vs Predicted Comparison', fontsize=20)

        ax[0].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)
        ax[0].set_title("True values", fontsize=18)
        ax[1].set_title("Predicted values", fontsize=18)
        plt.show()

if __name__ == "__main__":

    X_train = pickle.load(open('transformed_data/X_train', 'rb'))
    Y_train = pickle.load(open('transformed_data/Y_train', 'rb'))

    X_test = pickle.load(open('transformed_data/X_test', 'rb'))
    Y_test = pickle.load(open('transformed_data/Y_test', 'rb'))

   
    train = []
    for i, vector in enumerate(X_train):
        if Y_train[i] == 1:
            v = [vector] + ['pos']
        elif Y_train[i] == 0:
            v = [vector] + ['neg']
        train.append(v)
    
    df_train = pd.DataFrame(train)
    X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]

    test = []
    for i, vector in enumerate(X_test):
        if Y_test[i] == 1:
            v = [vector] + ['pos']
        elif Y_test[i] == 0:
            v = [vector] + ['neg']
        test.append(v)
    
    df_test = pd.DataFrame(test)
    X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

    model = NaiveBayesClassifier()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = model.accuracy(y_test, preds)

    print(classification_report(Y_test, preds))
    print('-'*55)
    print('Confusion Matrix\n')
    print(confusion_matrix(Y_test, preds))
    print(acc)
    