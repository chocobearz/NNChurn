import numpy as np
import pandas as pd

#random seeds for stochastic parts of neural network 
np.random.seed(10)
from tensorflow import set_random_seed
set_random_seed(15)

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding

cd = pd.read_csv("NNdata.csv")

X_train, X_test, y_train, y_test = train_test_split(
    cd.loc[:, cd.columns != 'churn'],
    cd["churn"],
    test_size=0.33,
    random_state=42
)

categorical_vars = ["internetservice", "contract", "paymentmethod"]

for categorical_var in categorical_vars :   
  model = Sequential()
  no_of_unique_cat  = X_train[categorical_var].nunique()
  embedding_size = min(np.ceil((no_of_unique_cat)/2), 50 )
  embedding_size = int(embedding_size)
  vocab  = no_of_unique_cat+1
  model.add( Embedding(no_of_unique_cat ,embedding_size, input_length = 1 ))
  model.add(Reshape(target_shape=(embedding_size,)))
  models.append( model )

print(model)
exit(0)
model_rest = Sequential()
model_rest.add(Dense(16, input_dim= 1 ))
models.append(model_rest)

full_model.add(Merge(models, mode='concat'))