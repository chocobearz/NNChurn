import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
from sklearn.model_selection import StratifiedKFold

cd = pd.read_csv("NNdata.csv")

X_train, X_test, y_train, y_test = train_test_split(
    cd.loc[:, cd.columns != 'churn'],
    cd["churn"],
    test_size=0.33,
    random_state=42
)

def build_embedding_network():
  inputs = []
  embeddings = []
  
  internetservice = Input(shape=(1,))
  embedding = Embedding(3, 2, input_length=1)(internetservice)
  embedding = Reshape(target_shape=(2,))(embedding)
  inputs.append(input_ps_ind_02_cat)
  embeddings.append(embedding)
  
  contract = Input(shape=(1,))
  embedding = Embedding(3, 2, input_length=1)(internetservice)
  embedding = Reshape(target_shape=(2,))(embedding)
  inputs.append(input_ps_ind_02_cat)
  embeddings.append(embedding)
  
  paymentmethod = Input(shape=(1,))
  embedding = Embedding(4, 2, input_length=1)(internetservice)
  embedding = Reshape(target_shape=(2,))(embedding)
  inputs.append(input_ps_ind_02_cat)
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

#network training
K = 8
runs_per_fold = 3
n_epochs = 15

cv_ginis = []
full_val_preds = np.zeros(np.shape(X_train)[0])
y_preds = np.zeros((np.shape(X_test)[0],K))

kfold = StratifiedKFold(n_splits = K, 
                            random_state = 231, 
                            shuffle = True)    

for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

    X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
    y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]
    
    X_test_f = X_test.copy()
    
    #upsampling adapted from kernel: 
    #https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
    pos = (pd.Series(y_train_f == 1))
    
    # Add positive examples
    X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
    y_train_f = pd.concat([y_train_f, y_train_f.loc[pos]], axis=0)
    
    # Shuffle data
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    y_train_f = y_train_f.iloc[idx]
    
    #track oof prediction for cv scores
    val_preds = 0
    
    for j in range(runs_per_fold):
    
        NN = build_embedding_network()
        NN.fit(proc_X_train_f, y_train_f.values, epochs=n_epochs, batch_size=4096, verbose=0)
   
        val_preds += NN.predict(proc_X_val_f)[:,0] / runs_per_fold
        y_preds[:,i] += NN.predict(proc_X_test_f)[:,0] / runs_per_fold
        
    full_val_preds[outf_ind] += val_preds
        
    cv_gini = gini_normalizedc(y_val_f.values, val_preds)
    cv_ginis.append(cv_gini)
    print ('\nFold %i prediction cv gini: %.5f\n' %(i,cv_gini))
    
print('Mean out of fold gini: %.5f' % np.mean(cv_ginis))
print('Full validation gini: %.5f' % gini_normalizedc(y_train.values, full_val_preds))

y_pred_final = np.mean(y_preds, axis=1)

df_sub = pd.DataFrame({'id' : df_test.id, 
                       'target' : y_pred_final},
                       columns = ['id','target'])
df_sub.to_csv('NN_EntityEmbed_10fold-sub.csv', index=False)

pd.DataFrame(full_val_preds).to_csv('NN_EntityEmbed_10fold-val_preds.csv',index=False)