import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding

cd = pd.read_csv("NNdata.csv")

X_train, X_test, y_train, y_test = train_test_split(
    cd.loc[:, cd.columns != 'churn'],
    cd["churn"],
    test_size=0.33,
    random_state=42
)

#def build_embedding_network():
    
inputs = []
embeddings = []

internetservice = Input(shape=(1,))

print(internetservice)
exit(0)
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


#x = Concatenate()(embeddings)
#    x = Dense(80, activation='relu')(x)
#    x = Dropout(.35)(x)
#    x = Dense(20, activation='relu')(x)
#    x = Dropout(.15)(x)
#    x = Dense(10, activation='relu')(x)
#    x = Dropout(.15)(x)
#    output = Dense(1, activation='sigmoid')(x)