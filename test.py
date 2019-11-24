import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
from sklearn.model_selection import StratifiedKFold

cd = pd.read_csv("oneHotBalance.csv")

X_train, X_test, y_train, y_test = train_test_split(
    cd.loc[:, cd.columns != 'churn'],
    cd["churn"],
    test_size=0.33,
    random_state=42
)

def build_embedding_network():


    #initialize constructor
    model = Sequential()

    #add input layer
    model.add(Dense(26, activation = 'relu', input_shape = (26,)))
    
    #add hidden layer
    model.add(Dense(20, activation = 'relu'))

    model.add(Dropout(.5))
    """
    #add hidden layer
    model.add(Dense(15, activation = 'relu'))

    model.add(Dropout(.3))
    
    #add hidden layer
    model.add(Dense(20, activation = 'relu'))

    model.add(Dropout(.4))

    #add hidden layer
    model.add(Dense(5, activation = 'relu'))

    model.add(Dropout(.1))
    """
    #add output layer
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

#If I have time will look at this lean embeddings
"""
def build_embedding_network():
  inputs = []
  embeddings = []
  
  internetservice = Input(shape=(1,))
  embedding = Embedding(3, 2, input_length=1)(internetservice)
  embedding = Reshape(target_shape=(2,))(embedding)
  inputs.append(internetservice)
  embeddings.append(embedding)
  
  contract = Input(shape=(1,))
  embedding = Embedding(3, 2, input_length=1)(contract)
  embedding = Reshape(target_shape=(2,))(embedding)
  inputs.append(contract)
  embeddings.append(embedding)
  
  paymentmethod = Input(shape=(1,))
  embedding = Embedding(4, 2, input_length=1)(paymentmethod)
  embedding = Reshape(target_shape=(2,))(embedding)
  inputs.append(paymentmethod)
  embeddings.append(embedding)
  
  input_numeric = Input(shape=(16,))
  embedding_numeric = Dense(16)(input_numeric) 
  inputs.append(input_numeric)
  embeddings.append(embedding_numeric)
  
  x = Concatenate()(embeddings)
  x = Dense(80, activation='relu')(x)
  x = Dropout(.35)(x)
  x = Dense(20, activation='relu')(x)
  x = Dropout(.15)(x)
  x = Dense(10, activation='relu')(x)
  x = Dropout(.15)(x)
  output = Dense(1, activation='sigmoid')(x)
  
  model = Model(inputs, output)
  model.compile(loss='binary_crossentropy', optimizer='adam')
  
  return model
"""

#network training
K = 2
runs_per_fold = 15
n_epochs = 5

y_preds = np.zeros((np.shape(X_test)[0],K))
scores = []

kfold = StratifiedKFold(n_splits = K, 
                            random_state = 231, 
                            shuffle = True)    

for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

    for j in range(runs_per_fold):
    
        NN = build_embedding_network()
        NN.fit(X_train, y_train.values, epochs=n_epochs, batch_size=40, verbose=0)
   
        y_preds[:,i] += NN.predict(X_test)[:,0] / runs_per_fold
        score = NN.evaluate(X_test, y_test, verbose = 1)
        scores.append(score)

y_pred_final = np.round(np.mean(y_preds, axis=1))

print(y_train)

print(y_pred_final)
print(y_test)

print(confusion_matrix(y_test, y_pred_final))
print(cohen_kappa_score(y_test, y_pred_final))
print(scores)
