import numpy as np
import pickle
from numpy import log, dot, e
from numpy.random import rand
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class LogisticRegression:
    
    def sigmoid(self, z): 
        return 1 / (1 + e**(-z)) - 0.00001
    
    def cost_function(self, X, y, weights):                 
        z = dot(X, weights)   
        sig = self.sigmoid(z)
        predict_1 = y * log(sig)
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))     
        return -sum(predict_1 + predict_0) / len(X)
    
    def fit(self, X, y, epochs=25, lr=0.05):        
        loss = []
        weights = rand(X.shape[1])
        N = len(X)
                 
        for i in range(epochs):
            print(i, epochs)        
            # Gradient Descent
            y_hat = self.sigmoid(dot(X, weights))
            weights -= lr * dot(X.T,  y_hat - y) / N            
            # Saving Progress
            l = self.cost_function(X, y, weights)
            loss.append(l) 
            print('Loss:', l)
            print()
        self.weights = weights
        self.loss = loss
    
    def predict(self, X):        
        # Predicting with sigmoid function
        z = dot(X, self.weights)
        # Returning binary result
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]

if __name__ == "__main__":

    X_train = pickle.load(open('transformed_data/X_train', 'rb'))
    Y_train = pickle.load(open('transformed_data/Y_train', 'rb'))

    X_test = pickle.load(open('transformed_data/X_test', 'rb'))
    Y_test = pickle.load(open('transformed_data/Y_test', 'rb'))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    scaler = MinMaxScaler() 
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train, epochs=1500, lr=0.1)

    y_pred = logreg.predict(X_test)
    print(classification_report(Y_test, y_pred))
    print('-'*55)
    print('Confusion Matrix\n')
    print(confusion_matrix(Y_test, y_pred))

    plt.style.use('seaborn-whitegrid')
    plt.rcParams['figure.dpi'] = 227
    plt.rcParams['figure.figsize'] = (16,5)
    plt.plot(logreg.loss)
    plt.title('Logistic Regression Training', fontSize=15)
    plt.xlabel('Epochs', fontSize=12)
    plt.ylabel('Loss', fontSize=12)
    plt.show()